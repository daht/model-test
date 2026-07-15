from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, status
from fastapi.responses import JSONResponse, Response
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_WEBSOCKET_HANDSHAKE_HEADERS = _HOP_BY_HOP_HEADERS | {
    "host",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
}
_INVALID_FORWARD_CLOSE_CODES = {1005, 1006, 1015}


@dataclass(frozen=True)
class BackendConfig:
    name: str
    http_url: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.http_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Backend {self.name!r} must use an http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f"Backend {self.name!r} URL must not contain credentials, query, or fragment")

    @property
    def websocket_url(self) -> str:
        parsed = urlsplit(self.http_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class GatewaySettings:
    backends: tuple[BackendConfig, ...]
    minimum_ready_backends: int = 2
    probe_interval_seconds: float = 1.0
    probe_timeout_seconds: float = 2.0
    upstream_open_timeout_seconds: float = 5.0
    upstream_close_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("At least one ASR gateway backend is required")
        if len({backend.name for backend in self.backends}) != len(self.backends):
            raise ValueError("ASR gateway backend names must be unique")
        if not 1 <= self.minimum_ready_backends <= len(self.backends):
            raise ValueError("Minimum ready backends must be between one and the backend count")
        for value in (
            self.probe_interval_seconds,
            self.probe_timeout_seconds,
            self.upstream_open_timeout_seconds,
            self.upstream_close_timeout_seconds,
        ):
            if value <= 0:
                raise ValueError("ASR gateway timeouts and intervals must be positive")

    @classmethod
    def from_environment(cls) -> "GatewaySettings":
        raw_backends = os.getenv(
            "ASR_GATEWAY_BACKENDS",
            "qwen-asr-backend-1=http://qwen-asr-backend-1:8000,"
            "qwen-asr-backend-2=http://qwen-asr-backend-2:8000",
        )
        backends = []
        for index, entry in enumerate(raw_backends.split(","), start=1):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                name, url = entry.split("=", 1)
            else:
                name, url = f"asr-{index}", entry
            backends.append(BackendConfig(name.strip(), url.strip().rstrip("/")))
        return cls(
            backends=tuple(backends),
            minimum_ready_backends=int(os.getenv("ASR_GATEWAY_MIN_READY_BACKENDS", "2")),
            probe_interval_seconds=float(os.getenv("ASR_GATEWAY_PROBE_INTERVAL_SECONDS", "1.0")),
            probe_timeout_seconds=float(os.getenv("ASR_GATEWAY_PROBE_TIMEOUT_SECONDS", "2.0")),
            upstream_open_timeout_seconds=float(
                os.getenv("ASR_GATEWAY_UPSTREAM_OPEN_TIMEOUT_SECONDS", "5.0")
            ),
            upstream_close_timeout_seconds=float(
                os.getenv("ASR_GATEWAY_UPSTREAM_CLOSE_TIMEOUT_SECONDS", "5.0")
            ),
        )


@dataclass
class _BackendState:
    config: BackendConfig
    ready: bool = False
    remote_active_streams: int = 0
    queue_depth: int = 0
    queued_audio_seconds: float = 0.0
    local_load: int = 0
    model: str | None = None
    backend: str | None = None
    detail: str | None = "not probed"

    @property
    def effective_load(self) -> int:
        # Upstream active_streams may already include gateway-owned connections.
        # max() fills probe lag without counting those connections twice.
        return max(self.remote_active_streams, self.local_load)


class BackendLease:
    def __init__(self, pool: "BackendPool", backend: BackendConfig) -> None:
        self._pool = pool
        self.backend = backend
        self._released = False
        self._release_lock = asyncio.Lock()

    async def release(self) -> None:
        async with self._release_lock:
            if self._released:
                return
            self._released = True
            await self._pool._release(self.backend.name)


class BackendPool:
    def __init__(
        self,
        settings: GatewaySettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._states = [_BackendState(config=backend) for backend in settings.backends]
        self._lock = asyncio.Lock()
        self._http_client = http_client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=None,
        )
        self._owns_http_client = http_client is None

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def refresh(self) -> None:
        await asyncio.gather(*(self._probe(state.config) for state in self._states))

    async def _probe(self, backend: BackendConfig) -> None:
        try:
            response = await self._http_client.get(
                f"{backend.http_url}/ready",
                timeout=self.settings.probe_timeout_seconds,
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("readiness payload is not an object")
            if response.status_code != 200 or payload.get("status") != "ready":
                await self.set_probe_result(
                    backend.name,
                    payload,
                    ready=False,
                    detail=f"upstream returned HTTP {response.status_code}",
                )
                return
            await self.set_probe_result(backend.name, payload)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            await self.set_probe_result(
                backend.name,
                {},
                ready=False,
                detail=f"probe failed ({type(exc).__name__})",
            )

    async def set_probe_result(
        self,
        backend_name: str,
        payload: dict[str, Any],
        *,
        ready: bool | None = None,
        detail: str | None = None,
    ) -> None:
        async with self._lock:
            state = self._state(backend_name)
            state.ready = payload.get("status") == "ready" if ready is None else ready
            state.remote_active_streams = _nonnegative_int(payload.get("active_streams", 0))
            state.queue_depth = _nonnegative_int(payload.get("queue_depth", 0))
            state.queued_audio_seconds = _nonnegative_float(
                payload.get("queued_audio_seconds", 0.0)
            )
            state.model = _optional_string(payload.get("model"))
            state.backend = _optional_string(payload.get("backend"))
            state.detail = detail if detail is not None else _optional_string(payload.get("detail"))

    async def reserve(self) -> BackendLease | None:
        async with self._lock:
            ready_states = [state for state in self._states if state.ready]
            if len(ready_states) < self.settings.minimum_ready_backends:
                return None
            state = min(
                ready_states,
                key=lambda candidate: (
                    candidate.effective_load,
                    candidate.queue_depth,
                    candidate.queued_audio_seconds,
                    self._states.index(candidate),
                ),
            )
            state.local_load += 1
            return BackendLease(self, state.config)

    async def _release(self, backend_name: str) -> None:
        async with self._lock:
            state = self._state(backend_name)
            if state.local_load <= 0:
                logger.error("asr_gateway_accounting_underflow backend=%s", backend_name)
                state.ready = False
                state.detail = "gateway accounting underflow"
                return
            state.local_load -= 1

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            ready_count = sum(state.ready for state in self._states)
            ready_states = [state for state in self._states if state.ready]
            return {
                "status": (
                    "ready"
                    if ready_count >= self.settings.minimum_ready_backends
                    else "not_ready"
                ),
                "model": next((state.model for state in ready_states if state.model), None),
                "backend": next((state.backend for state in ready_states if state.backend), None),
                "active_streams": sum(
                    max(state.remote_active_streams, state.local_load)
                    for state in ready_states
                ),
                "queue_depth": sum(state.queue_depth for state in ready_states),
                "queued_audio_seconds": sum(
                    state.queued_audio_seconds for state in ready_states
                ),
                "ready_backend_count": ready_count,
                "minimum_ready_backends": self.settings.minimum_ready_backends,
                "backends": [
                    {
                        "name": state.config.name,
                        "ready": state.ready,
                        "remote_active_streams": state.remote_active_streams,
                        "local_load": state.local_load,
                        "effective_load": state.effective_load,
                        "queue_depth": state.queue_depth,
                        "queued_audio_seconds": state.queued_audio_seconds,
                        "detail": state.detail,
                    }
                    for state in self._states
                ],
            }

    def _state(self, backend_name: str) -> _BackendState:
        for state in self._states:
            if state.config.name == backend_name:
                return state
        raise KeyError(backend_name)


_runtime_pool: BackendPool | None = None
_test_pool: BackendPool | None = None


def set_gateway_pool_for_tests(pool: BackendPool | None) -> None:
    global _test_pool
    _test_pool = pool


def get_gateway_pool() -> BackendPool:
    pool = _test_pool or _runtime_pool
    if pool is None:
        raise RuntimeError("ASR gateway backend pool is not initialized")
    return pool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _runtime_pool
    if _test_pool is not None:
        yield
        return
    pool = BackendPool(GatewaySettings.from_environment())
    _runtime_pool = pool
    stop = asyncio.Event()
    probe_task: asyncio.Task[None] | None = None
    try:
        await pool.refresh()
        probe_task = asyncio.create_task(_probe_loop(pool, stop), name="asr-gateway-probes")
        yield
    finally:
        stop.set()
        if probe_task is not None:
            probe_task.cancel()
            with suppress(asyncio.CancelledError):
                await probe_task
        await pool.close()
        _runtime_pool = None


async def _probe_loop(pool: BackendPool, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=pool.settings.probe_interval_seconds)
        except TimeoutError:
            await pool.refresh()


app = FastAPI(
    title="ASR Multi-Pod Gateway",
    version="0.1.0",
    description="Experimental sticky proxy for independent Qwen ASR backends.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    snapshot = await get_gateway_pool().snapshot()
    return {
        "status": "ok",
        "service": "asr-gateway",
        "ready_backend_count": snapshot["ready_backend_count"],
    }


@app.get("/ready")
async def ready() -> Response:
    snapshot = await get_gateway_pool().snapshot()
    code = 200 if snapshot["status"] == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(snapshot, status_code=code)


@app.api_route(
    "/v1/{upstream_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_http(request: Request, upstream_path: str) -> Response:
    pool = get_gateway_pool()
    lease = await pool.reserve()
    if lease is None:
        return JSONResponse(
            {"detail": {"code": "not_ready", "message": "ASR gateway has insufficient ready backends"}},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    try:
        query = request.url.query
        url = f"{lease.backend.http_url}/v1/{upstream_path}"
        if query:
            url = f"{url}?{query}"
        headers = _forward_http_headers(request.headers.items())
        try:
            upstream = await pool._http_client.request(
                request.method,
                url,
                headers=headers,
                content=request.stream(),
                follow_redirects=False,
            )
        except httpx.HTTPError:
            logger.warning("asr_gateway_http_upstream_failed backend=%s", lease.backend.name)
            return JSONResponse(
                {"detail": {"code": "upstream_unavailable", "message": "Selected ASR backend is unavailable"}},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() not in {"content-length", "content-encoding"}
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=None,
        )
    finally:
        await lease.release()


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket) -> None:
    await proxy_websocket(
        websocket,
        get_gateway_pool(),
        path=websocket.url.path,
        query=websocket.url.query,
        headers=websocket.headers.items(),
    )


async def proxy_websocket(
    websocket: Any,
    pool: BackendPool,
    *,
    path: str = "/v1/transcribe/stream",
    query: str = "",
    headers: Iterable[tuple[str, str]] = (),
    connect: Callable[..., AsyncContextManager[Any]] = websockets.connect,
) -> None:
    lease = await pool.reserve()
    await websocket.accept()
    if lease is None:
        await websocket.close(code=1013, reason="ASR gateway is not ready")
        return

    upstream_url = f"{lease.backend.websocket_url}{path}"
    if query:
        upstream_url = f"{upstream_url}?{query}"
    try:
        async with connect(
            upstream_url,
            additional_headers=_forward_websocket_headers(headers),
            open_timeout=pool.settings.upstream_open_timeout_seconds,
            close_timeout=pool.settings.upstream_close_timeout_seconds,
            compression=None,
            proxy=None,
            max_size=None,
            max_queue=4,
        ) as upstream:
            await _run_websocket_proxy(websocket, upstream)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("asr_gateway_websocket_upstream_failed backend=%s", lease.backend.name)
        with suppress(Exception):
            await websocket.close(code=1011, reason="Upstream proxy failure")
    finally:
        await lease.release()


async def _run_websocket_proxy(client: Any, upstream: Any) -> None:
    client_task = asyncio.create_task(_client_to_upstream(client, upstream))
    upstream_task = asyncio.create_task(_upstream_to_client(upstream, client))
    tasks = {client_task, upstream_task}
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
        if upstream_task in done:
            close_code, close_reason = upstream_task.result()
            await client.close(code=close_code, reason=close_reason)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _client_to_upstream(client: Any, upstream: Any) -> None:
    while True:
        message = await client.receive()
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            code = _forward_close_code(message.get("code"), fallback=1000)
            await upstream.close(code=code)
            return
        if message_type != "websocket.receive":
            continue
        if message.get("bytes") is not None:
            await upstream.send(message["bytes"])
        elif message.get("text") is not None:
            await upstream.send(message["text"])


async def _upstream_to_client(upstream: Any, client: Any) -> tuple[int, str]:
    while True:
        try:
            message = await upstream.recv()
        except ConnectionClosed as exc:
            close = exc.rcvd or exc.sent
            if close is None:
                return 1011, ""
            return _forward_close_code(close.code, fallback=1011), close.reason or ""
        if isinstance(message, str):
            await client.send_text(message)
        else:
            await client.send_bytes(bytes(message))


def _forward_http_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }


def _forward_websocket_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _WEBSOCKET_HANDSHAKE_HEADERS
    }


def _forward_close_code(value: Any, *, fallback: int) -> int:
    if not isinstance(value, int) or value < 1000 or value > 4999:
        return fallback
    if value in _INVALID_FORWARD_CLOSE_CODES:
        return fallback
    return value


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
