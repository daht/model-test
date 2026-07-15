import asyncio
import math
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from app import asr_gateway


def _settings(**overrides):
    values = {
        "backends": (
            asr_gateway.BackendConfig("asr-1", "http://asr-1:8000"),
            asr_gateway.BackendConfig("asr-2", "http://asr-2:8000"),
        ),
        "minimum_ready_backends": 2,
        "probe_interval_seconds": 1.0,
        "probe_timeout_seconds": 0.25,
        "upstream_open_timeout_seconds": 0.25,
        "upstream_close_timeout_seconds": 0.25,
    }
    values.update(overrides)
    return asr_gateway.GatewaySettings(**values)


def _ready_payload(active_streams=0):
    return {
        "status": "ready",
        "model": "Qwen3-ASR-1.7B",
        "backend": "qwen_vllm",
        "active_streams": active_streams,
        "queue_depth": 0,
        "queued_audio_seconds": 0.0,
        "detail": None,
    }


def test_atomic_reservations_distribute_equal_simultaneous_admissions():
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        await pool.set_probe_result("asr-1", _ready_payload())
        await pool.set_probe_result("asr-2", _ready_payload())
        start = asyncio.Event()

        async def reserve():
            await start.wait()
            return await pool.reserve()

        tasks = [asyncio.create_task(reserve()) for _ in range(2)]
        start.set()
        leases = await asyncio.gather(*tasks)
        selected = [lease.backend.name for lease in leases]
        snapshot = await pool.snapshot()
        for lease in leases:
            await lease.release()
        released = await pool.snapshot()
        return selected, snapshot, released

    selected, snapshot, released = asyncio.run(scenario())

    assert set(selected) == {"asr-1", "asr-2"}
    assert [item["local_load"] for item in snapshot["backends"]] == [1, 1]
    assert [item["local_load"] for item in released["backends"]] == [0, 0]


def test_unready_and_unreachable_backends_are_excluded_and_readiness_fails_closed():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "asr-1":
            return httpx.Response(200, json=_ready_payload(active_streams=3))
        raise httpx.ConnectError("unreachable", request=request)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pool = asr_gateway.BackendPool(_settings(), http_client=client)
        await pool.refresh()
        lease = await pool.reserve()
        snapshot = await pool.snapshot()
        await client.aclose()
        return lease, snapshot

    lease, snapshot = asyncio.run(scenario())

    assert lease is None
    assert snapshot["status"] == "not_ready"
    assert snapshot["ready_backend_count"] == 1
    assert snapshot["backends"][1]["ready"] is False
    assert "unreachable" not in snapshot["backends"][1]["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in _ready_payload().items() if key != "active_streams"},
        {**_ready_payload(), "queue_depth": -1},
        {**_ready_payload(), "queued_audio_seconds": float("nan")},
        {**_ready_payload(), "active_streams": "1e10000"},
        {**_ready_payload(), "queue_depth": 10**100},
        {**_ready_payload(), "queued_audio_seconds": float("inf")},
        {**_ready_payload(), "model": ""},
        {**_ready_payload(), "backend": None},
    ],
    ids=(
        "missing-field",
        "negative",
        "nan",
        "overflow",
        "unbounded-integer",
        "infinity",
        "empty-model",
        "invalid-backend",
    ),
)
def test_malformed_ready_probe_fails_closed_and_clears_stale_state(payload):
    asr_1_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal asr_1_calls
        if request.url.host == "asr-2":
            return httpx.Response(503, json={"status": "not_ready"})
        asr_1_calls += 1
        response_payload = _ready_payload(active_streams=4) if asr_1_calls == 1 else payload
        return httpx.Response(200, json=response_payload)

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pool = asr_gateway.BackendPool(
            _settings(minimum_ready_backends=1), http_client=client
        )
        await pool.refresh()
        before = await pool.snapshot()
        await pool.refresh()
        after = await pool.snapshot()
        await client.aclose()
        return before, after

    before, after = asyncio.run(scenario())

    assert before["status"] == "ready"
    assert after["status"] == "not_ready"
    assert after["ready_backend_count"] == 0
    assert after["backends"][0]["remote_active_streams"] == 0
    assert after["backends"][0]["queue_depth"] == 0
    assert after["backends"][0]["queued_audio_seconds"] == 0.0


def test_scheduler_excludes_explicitly_unready_backend():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "asr-1":
            return httpx.Response(200, json=_ready_payload(active_streams=9))
        return httpx.Response(
            503,
            json={"status": "not_ready", "active_streams": 0, "detail": "loading"},
        )

    async def scenario():
        settings = _settings(minimum_ready_backends=1)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pool = asr_gateway.BackendPool(settings, http_client=client)
        await pool.refresh()
        lease = await pool.reserve()
        snapshot = await pool.snapshot()
        await lease.release()
        await client.aclose()
        return lease.backend.name, snapshot

    selected, snapshot = asyncio.run(scenario())

    assert selected == "asr-1"
    assert snapshot["status"] == "ready"
    assert snapshot["backends"][1]["ready"] is False


def test_gateway_ready_endpoint_fails_closed_below_minimum():
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        await pool.set_probe_result("asr-1", _ready_payload())
        asr_gateway.set_gateway_pool_for_tests(pool)

    asyncio.run(scenario())
    try:
        with TestClient(asr_gateway.app) as client:
            response = client.get("/ready")
    finally:
        asr_gateway.set_gateway_pool_for_tests(None)

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_http_contract_forwards_only_stream_info_and_rejects_other_v1_paths():
    seen = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "authorization": request.headers.get("authorization"),
                "body": await request.aread(),
            }
        )
        if request.url.path.endswith("stream-info"):
            return httpx.Response(
                200,
                json={
                    "protocol_version": 2,
                    "websocket_url": "/v1/transcribe/stream",
                },
            )
        raise AssertionError("unsupported HTTP path reached an upstream")

    async def setup():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        pool = asr_gateway.BackendPool(_settings(), http_client=client)
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        return pool, client

    pool, upstream_client = asyncio.run(setup())
    asr_gateway.set_gateway_pool_for_tests(pool)
    try:
        with TestClient(asr_gateway.app) as client:
            info = client.get("/v1/transcribe/stream-info?detail=1")
            posted = client.post(
                "/v1/transcribe",
                headers={"authorization": "Bearer unit-test-token"},
                content=b"language=zh",
            )
    finally:
        asr_gateway.set_gateway_pool_for_tests(None)
        asyncio.run(upstream_client.aclose())

    assert info.status_code == 200
    assert info.json()["protocol_version"] == 2
    assert posted.status_code == 404
    assert posted.json()["detail"]["code"] == "unsupported_gateway_path"
    assert seen[0]["query"] == b"detail=1"
    assert len(seen) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "setting_name",
    [
        "probe_interval_seconds",
        "probe_timeout_seconds",
        "upstream_open_timeout_seconds",
        "upstream_close_timeout_seconds",
    ],
)
def test_gateway_settings_reject_nonfinite_timeouts_and_intervals(setting_name, value):
    assert not math.isfinite(value)
    with pytest.raises(ValueError, match="finite and positive"):
        _settings(**{setting_name: value})


class _ClientWebSocket:
    def __init__(self, messages=None):
        self.messages = asyncio.Queue()
        for message in messages or []:
            self.messages.put_nowait(message)
        self.accepted = False
        self.sent = []
        self.closes = []

    async def accept(self):
        self.accepted = True

    async def receive(self):
        return await self.messages.get()

    async def send_text(self, value):
        self.sent.append(("text", value))

    async def send_bytes(self, value):
        self.sent.append(("bytes", value))

    async def close(self, code=1000, reason=None):
        self.closes.append((code, reason))


class _FailingAcceptClient(_ClientWebSocket):
    def __init__(self, error):
        super().__init__()
        self.error = error

    async def accept(self):
        raise self.error


class _UpstreamWebSocket:
    def __init__(self, incoming=(), *, send_error=None, receive_after_sends=0):
        self.incoming = asyncio.Queue()
        for item in incoming:
            self.incoming.put_nowait(item)
        self.sent = []
        self.send_error = send_error
        self.receive_after_sends = receive_after_sends
        self.entered = asyncio.Event()
        self.ready_to_receive = asyncio.Event()
        if receive_after_sends == 0:
            self.ready_to_receive.set()
        self.closed = False

    async def __aenter__(self):
        self.entered.set()
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def recv(self):
        await self.ready_to_receive.wait()
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, value):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(value)
        if len(self.sent) >= self.receive_after_sends:
            self.ready_to_receive.set()

    async def close(self, code=1000, reason=""):
        self.closed = True


def _closed(code, reason=""):
    close = Close(code, reason)
    if code == 1000:
        return ConnectionClosedOK(close, close, True)
    return ConnectionClosedError(close, close, True)


@pytest.mark.parametrize("close_code", [1000, 1013])
def test_websocket_preserves_frames_and_upstream_terminal_close_code(close_code):
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        await pool.set_probe_result("asr-1", _ready_payload())
        await pool.set_probe_result("asr-2", _ready_payload(active_streams=1))
        client = _ClientWebSocket(
            [
                {"type": "websocket.receive", "text": '{"type":"start"}'},
                {"type": "websocket.receive", "bytes": b"\x01\x02"},
            ]
        )
        upstream = _UpstreamWebSocket(
            ["{\"type\":\"ready\",\"sequence\":1}", b"binary-result", _closed(close_code)],
            receive_after_sends=2,
        )
        connect_call = {}

        @asynccontextmanager
        async def connect(*args, **kwargs):
            connect_call.update({"args": args, "kwargs": kwargs})
            async with upstream as connection:
                yield connection

        await asr_gateway.proxy_websocket(
            client,
            pool,
            headers=(
                ("authorization", "Bearer unit-test-token"),
                ("host", "gateway.test"),
                ("sec-websocket-key", "handshake-only"),
            ),
            connect=connect,
        )
        return client, upstream, connect_call, await pool.snapshot()

    client, upstream, connect_call, snapshot = asyncio.run(scenario())

    assert upstream.sent[:2] == ['{"type":"start"}', b"\x01\x02"]
    assert client.sent == [
        ("text", '{"type":"ready","sequence":1}'),
        ("bytes", b"binary-result"),
    ]
    assert client.closes == [(close_code, "")]
    assert connect_call["kwargs"]["additional_headers"] == {
        "authorization": "Bearer unit-test-token"
    }
    assert [item["local_load"] for item in snapshot["backends"]] == [0, 0]


def test_websocket_proxy_failure_and_cancellation_release_accounting_once():
    async def failure_scenario():
        pool = asr_gateway.BackendPool(_settings())
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        client = _ClientWebSocket(
            [{"type": "websocket.receive", "text": "request"}]
        )
        upstream = _UpstreamWebSocket(send_error=RuntimeError("send failed"))

        @asynccontextmanager
        async def connect(*_args, **_kwargs):
            async with upstream as connection:
                yield connection

        await asr_gateway.proxy_websocket(client, pool, connect=connect)
        return client, await pool.snapshot()

    async def cancellation_scenario():
        pool = asr_gateway.BackendPool(_settings())
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        client = _ClientWebSocket()
        upstream = _UpstreamWebSocket()

        @asynccontextmanager
        async def connect(*_args, **_kwargs):
            async with upstream as connection:
                yield connection

        task = asyncio.create_task(
            asr_gateway.proxy_websocket(client, pool, connect=connect)
        )
        await upstream.entered.wait()
        during = await pool.snapshot()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        after = await pool.snapshot()
        return during, after

    failed_client, after_failure = asyncio.run(failure_scenario())
    during_cancel, after_cancel = asyncio.run(cancellation_scenario())

    assert failed_client.closes == [(1011, "Upstream proxy failure")]
    assert sum(item["local_load"] for item in after_failure["backends"]) == 0
    assert all(item["ready"] for item in after_failure["backends"])
    assert sum(item["local_load"] for item in during_cancel["backends"]) == 1
    assert sum(item["local_load"] for item in after_cancel["backends"]) == 0
    assert all(item["ready"] for item in after_cancel["backends"])


@pytest.mark.parametrize(
    ("error", "propagates"),
    [(RuntimeError("accept failed"), False), (asyncio.CancelledError(), True)],
)
def test_websocket_accept_failure_or_cancellation_releases_reservation(error, propagates):
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        client = _FailingAcceptClient(error)

        if propagates:
            with pytest.raises(type(error)):
                await asr_gateway.proxy_websocket(client, pool)
        else:
            await asr_gateway.proxy_websocket(client, pool)
        return await pool.snapshot()

    snapshot = asyncio.run(scenario())

    assert sum(item["local_load"] for item in snapshot["backends"]) == 0


def test_simultaneous_upstream_terminal_close_wins_over_client_forwarding_failure():
    class Barrier:
        def __init__(self):
            self.arrivals = 0
            self.release = asyncio.Event()

        async def wait(self):
            self.arrivals += 1
            if self.arrivals == 2:
                self.release.set()
            await self.release.wait()

    class RacingClient(_ClientWebSocket):
        async def receive(self):
            await barrier.wait()
            raise RuntimeError("client receive failed")

    class RacingUpstream(_UpstreamWebSocket):
        async def recv(self):
            await barrier.wait()
            raise _closed(1013, "retry elsewhere")

    async def scenario():
        nonlocal barrier
        barrier = Barrier()
        client = RacingClient()
        upstream = RacingUpstream()
        await asr_gateway._run_websocket_proxy(client, upstream)
        return client

    barrier = None
    client = asyncio.run(scenario())

    assert client.closes == [(1013, "retry elsewhere")]


def test_upstream_connect_failure_releases_accounting_once():
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        client = _ClientWebSocket()

        @asynccontextmanager
        async def connect(*_args, **_kwargs):
            raise OSError("connect failed")
            yield

        await asr_gateway.proxy_websocket(client, pool, connect=connect)
        return client, await pool.snapshot()

    client, snapshot = asyncio.run(scenario())

    assert client.closes == [(1011, "Upstream proxy failure")]
    assert sum(item["local_load"] for item in snapshot["backends"]) == 0


def test_client_disconnect_releases_accounting_once():
    async def scenario():
        pool = asr_gateway.BackendPool(_settings())
        for name in ("asr-1", "asr-2"):
            await pool.set_probe_result(name, _ready_payload())
        client = _ClientWebSocket(
            [{"type": "websocket.disconnect", "code": 1000}]
        )
        upstream = _UpstreamWebSocket()

        @asynccontextmanager
        async def connect(*_args, **_kwargs):
            async with upstream as connection:
                yield connection

        await asr_gateway.proxy_websocket(client, pool, connect=connect)
        return upstream, await pool.snapshot()

    upstream, snapshot = asyncio.run(scenario())

    assert upstream.closed is True
    assert sum(item["local_load"] for item in snapshot["backends"]) == 0


def test_gateway_image_is_slim_cpu_only_and_dependency_free():
    dockerfile_path = Path("Dockerfile.asr-gateway")

    assert dockerfile_path.is_file()
    dockerfile = dockerfile_path.read_text()
    assert "FROM python:3.11-slim" in dockerfile
    assert "# syntax=docker/dockerfile:1.7" in dockerfile
    assert "RUN --mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "requirements-asr-gateway.txt" in dockerfile
    assert "COPY app ./app" in dockerfile
    assert "app.asr_gateway:app" in dockerfile
    for forbidden in (
        "torch",
        "vllm",
        "qwen-asr",
        "cuda",
        "transformers",
        "accelerate",
        "silero",
        "onnx",
        "ffmpeg",
        "libsndfile",
        "model",
        "audio",
    ):
        assert forbidden not in dockerfile.lower()


def test_gateway_requirements_are_exactly_the_pinned_http_stack():
    requirements_path = Path("requirements-asr-gateway.txt")

    assert requirements_path.is_file()
    requirements = requirements_path.read_text().splitlines()
    assert requirements == [
        "fastapi==0.139.0",
        "uvicorn[standard]==0.38.0",
        "httpx==0.28.1",
        "websockets==16.0",
    ]


def test_multipod_compose_topology_is_explicit_and_gpu_gateway_free():
    compose = yaml.safe_load(Path("docker-compose.asr-multipod.yml").read_text())
    services = compose["services"]

    assert set(services) == {"qwen-asr-backend-1", "qwen-asr-backend-2", "asr-gateway"}
    for name in ("qwen-asr-backend-1", "qwen-asr-backend-2"):
        service = services[name]
        assert service["command"][service["command"].index("--workers") + 1] == "1"
        assert service["environment"]["ASR_VLLM_GPU_MEMORY_UTILIZATION"] == (
            "${ASR_MULTIPOD_GPU_MEMORY_UTILIZATION:-0.35}"
        )
        assert any(volume.endswith("/models:ro") for volume in service["volumes"])
        assert service["gpus"] == "all"

    backend_1 = services["qwen-asr-backend-1"]
    backend_2 = services["qwen-asr-backend-2"]
    assert backend_1["build"] == {"context": ".", "dockerfile": "Dockerfile.asr"}
    assert backend_1["image"] == "qwen-asr-api:latest"
    assert "build" not in backend_2
    assert backend_2["image"] == "qwen-asr-api:latest"
    assert backend_2["pull_policy"] == "never"
    assert backend_2["depends_on"] == {
        "qwen-asr-backend-1": {"condition": "service_healthy"}
    }
    assert sum("build" in services[name] for name in services) == 2

    gateway = services["asr-gateway"]
    assert gateway["build"] == {
        "context": ".",
        "dockerfile": "Dockerfile.asr-gateway",
    }
    assert gateway["image"] == "qwen-asr-gateway:latest"
    assert gateway["ports"] == ["8002:8000"]
    assert "gpus" not in gateway
    assert "env_file" not in gateway
    assert gateway["environment"]["ASR_GATEWAY_MIN_READY_BACKENDS"] == "2"
    assert set(gateway["depends_on"]) == {
        "qwen-asr-backend-1",
        "qwen-asr-backend-2",
    }
    serialized = Path("docker-compose.asr-multipod.yml").read_text().lower()
    assert "api_key" not in serialized
