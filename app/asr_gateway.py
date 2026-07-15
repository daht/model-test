from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.asr_gateway_backends import BackendLease, BackendRegistry, ResultMode
from app.asr_gateway_metrics import GatewayMetrics, gateway_readiness
from app.asr_gateway_protocol import ProtocolSession, StartCommand, parse_client_command
from app.asr_gateway_scheduler import BatchKey, GatewayScheduler, InferenceJob, InferenceResult
from app.asr_gateway_sessions import GatewaySession, SessionManager, TerminalState
from app.config import Settings, get_settings


@dataclass
class _SessionContext:
    session: GatewaySession
    lease: BackendLease
    protocol: ProtocolSession
    events: asyncio.Queue[dict[str, Any]]
    idle: asyncio.Event


class GatewayRuntime:
    def __init__(self, settings: Settings, adapters: Mapping[str, Any]) -> None:
        self.settings = settings
        self.adapters = dict(adapters)
        self.registry = BackendRegistry()
        self.sessions = SessionManager(max_sessions=settings.asr_gateway_max_active_sessions)
        self.metrics = GatewayMetrics()
        self._contexts: dict[str, _SessionContext] = {}
        self.scheduler = GatewayScheduler(
            self.adapters,
            max_wait_seconds=settings.asr_gateway_schedule_max_wait_ms / 1000,
            max_ready_jobs=settings.asr_gateway_max_ready_jobs,
            max_queued_samples=round(settings.asr_gateway_max_queued_audio_seconds * 16_000),
            cleanup=self._cleanup_job,
            publish=self._publish_result,
        )
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        try:
            for worker_id, adapter in self.adapters.items():
                await adapter.warmup()
                if adapter.capabilities.worker_id != worker_id:
                    raise ValueError("adapter map key must equal capability worker_id")
                await self.registry.register(adapter.capabilities)
                await self.registry.mark_ready(worker_id, True)
            await self.scheduler.start()
            self._started = True
        except Exception:
            for adapter in self.adapters.values():
                with suppress(Exception):
                    await adapter.close()
            raise

    async def close(self) -> None:
        for session_id in list(self._contexts):
            await self.abort(session_id)
        await self.scheduler.close()
        for adapter in self.adapters.values():
            await adapter.close()
        self._started = False

    async def open_session(self, session_id: str, *, language: str, options: dict[str, Any]) -> GatewaySession:
        lease = await self.registry.acquire()
        worker_id = lease.worker_id
        adapter = self.adapters[worker_id]
        try:
            backend_id = await adapter.open_session(session_id, language=language, **options)
            session = self.sessions.create(
                session_id, worker_id, backend_id, sample_rate=16_000,
                max_buffer_samples=round(self.settings.asr_gateway_max_session_buffer_seconds * 16_000),
                language=language, options=options,
            )
        except Exception:
            await lease.release()
            raise
        idle = asyncio.Event(); idle.set()
        self._contexts[session_id] = _SessionContext(
            session, lease, ProtocolSession(sample_rate=16_000), asyncio.Queue(), idle
        )
        return session

    async def ingest(self, session: GatewaySession, pcm: bytes, *, force: bool = False) -> bool:
        if pcm:
            session.append_pcm(pcm)
        return self._schedule_next(session, force=force)

    def _schedule_next(self, session: GatewaySession, *, force: bool = False) -> bool:
        preferred = round(self.settings.asr_gateway_default_update_ms / 1000 * session.sample_rate)
        caps = self.adapters[session.selected_worker_id].capabilities
        count = session.ready_samples(preferred=preferred, maximum=caps.max_input_samples, force=force)
        if count <= 0:
            return False
        reservation = session.reserve(count, final=session.finish_requested)
        ctx = self._contexts[session.session_id]
        ctx.idle.clear()
        bucket = min(15, max(0, count.bit_length() - 1))
        options = session.options
        job = InferenceJob(
            job_id=f"{session.session_id}:{session.generation}:{reservation.job_sequence}",
            session_id=session.session_id, generation=session.generation,
            job_sequence=reservation.job_sequence, worker_id=session.selected_worker_id,
            backend_session_id=session.backend_session_id,
            start_sample=reservation.chunk.start_sample, end_sample=reservation.chunk.end_sample,
            pcm=reservation.chunk.pcm,
            deadline=self.scheduler.clock() + self.settings.asr_gateway_schedule_max_wait_ms / 1000,
            batch_key=BatchKey(
                session.selected_worker_id, caps.model_revision, session.language,
                str(options.get("task", "transcribe")), bool(options.get("timestamps", False)),
                str(hash(str(options.get("prompt", "")))),
                str(hash(json.dumps(options, sort_keys=True, default=str))),
                "pcm_s16le", bucket,
            ),
            final=reservation.chunk.final,
        )
        try:
            self.scheduler.enqueue(job)
        except Exception:
            session.rollback(reservation.job_sequence)
            ctx.idle.set()
            raise
        return True

    async def wait_idle(self, session_id: str) -> None:
        await self._contexts[session_id].idle.wait()

    async def segment(self, session_id: str) -> list[dict[str, Any]]:
        ctx = self._contexts[session_id]
        await self.ingest(ctx.session, b"", force=True)
        await ctx.idle.wait()
        result = await self.adapters[ctx.session.selected_worker_id].finish_segment(session_id)
        if result.text:
            ctx.protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text=ctx.protocol.state.confirmed_text + result.text)
        return ctx.protocol.segment()

    async def finish(self, session_id: str) -> dict[str, Any]:
        ctx = self._contexts[session_id]
        ctx.session.request_finish()
        await self.ingest(ctx.session, b"", force=True)
        await ctx.idle.wait()
        result = await self.adapters[ctx.session.selected_worker_id].finish_session(session_id)
        if result.text:
            ctx.protocol.apply_result(ResultMode.CUMULATIVE_SNAPSHOT, text=ctx.protocol.state.confirmed_text + result.text)
        event = ctx.protocol.final()
        ctx.session.succeed()
        await self._release_session(session_id)
        return event

    async def abort(self, session_id: str) -> None:
        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        self.scheduler.cancel_session(session_id, generation=ctx.session.generation)
        ctx.session.abort()
        with suppress(Exception):
            await self.adapters[ctx.session.selected_worker_id].abort_session(session_id)
        await self._release_session(session_id)

    async def _release_session(self, session_id: str) -> None:
        ctx = self._contexts.pop(session_id, None)
        self.sessions.close(session_id)
        if ctx is not None:
            await ctx.lease.release()

    async def _cleanup_job(self, job: InferenceJob) -> None:
        ctx = self._contexts.get(job.session_id)
        if ctx is None or ctx.session.generation != job.generation:
            return
        ctx.session.acknowledge(job.job_sequence, generation=job.generation)
        if not self._schedule_next(ctx.session, force=ctx.session.finish_requested):
            ctx.idle.set()
        self._update_gauges()

    async def _publish_result(self, result: InferenceResult) -> None:
        ctx = self._contexts.get(result.session_id)
        if ctx is None:
            return
        if result.error:
            ctx.session.fail()
            await ctx.events.put(ctx.protocol.error(RuntimeError(result.error)))
            return
        mode = self.adapters[result.worker_id].capabilities.result_mode
        events = ctx.protocol.apply_result(
            mode, text=result.text, confirmed_text=result.confirmed_text,
            tail_text=result.tail_text, decoded_samples=result.end_sample - result.start_sample,
            segment_id=result.job_sequence,
        )
        for event in events:
            await ctx.events.put(event)

    def event_queue(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        return self._contexts[session_id].events

    def _update_gauges(self) -> None:
        scheduler = self.scheduler.snapshot()
        self.metrics.set_gauges(
            active_sessions=self.sessions.snapshot()["active_sessions"],
            ready_depth=scheduler["ready_depth"], queued_samples=scheduler["queued_samples"], sample_rate=16_000,
        )


def _authorized(value: str | None, settings: Settings) -> bool:
    return bool(value and value == settings.api_key)


def _default_runtime() -> GatewayRuntime:
    from app.asr import create_asr_transcriber
    from app.asr_gateway_local_adapter import LocalCoordinatorAdapter
    from app.asr_inference import ASRInferenceCoordinator

    settings = get_settings()
    adapter = LocalCoordinatorAdapter(
        lambda: ASRInferenceCoordinator(settings, lambda: create_asr_transcriber(settings)),
        worker_id="local", model_id=settings.asr_model_id,
        model_revision=settings.asr_model_name, gpu_id=settings.asr_device,
        session_capacity=settings.asr_max_active_streams,
    )
    return GatewayRuntime(settings, {"local": adapter})


def create_app(*, runtime: GatewayRuntime | None = None) -> FastAPI:
    holder: dict[str, GatewayRuntime] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        selected = runtime or _default_runtime()
        holder["runtime"] = selected
        await selected.start()
        try:
            yield
        finally:
            await selected.close()

    app = FastAPI(title="Semantic ASR Gateway", version="1.0", lifespan=lifespan)

    def current() -> GatewayRuntime:
        return holder["runtime"]

    def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not _authorized(x_api_key, current().settings):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        value = await gateway_readiness(current().registry)
        payload = {"status": "ready" if value else "not_ready"}
        return payload if value else JSONResponse(status_code=503, content=payload)

    @app.get("/v1/transcribe/stream-info", dependencies=[])
    async def stream_info(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        if not _authorized(x_api_key, current().settings):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return {"protocol_version": 2, "websocket_url": "/v1/transcribe/stream", "format": "pcm_s16le", "sample_rate": 16000, "channels": 1}

    @app.get("/v1/asr/backends")
    async def backends(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        return await current().registry.snapshot()

    @app.get("/v1/asr/metrics")
    async def metrics(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        require_key(x_api_key)
        return current().metrics.snapshot()

    @app.websocket("/v1/transcribe/stream")
    async def stream(websocket: WebSocket) -> None:
        runtime_value = current()
        if not _authorized(websocket.headers.get("x-api-key"), runtime_value.settings):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session_id = uuid.uuid4().hex
        protocol: ProtocolSession | None = None
        sender: asyncio.Task[None] | None = None
        terminal_sent = False
        try:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=runtime_value.settings.asr_start_timeout_seconds)
                command = parse_client_command(message)
                if not isinstance(command, StartCommand):
                    raise ValueError("first command must be start")
                session = await runtime_value.open_session(session_id, language=command.language, options=command.options)
                protocol = runtime_value._contexts[session_id].protocol
            except RuntimeError as exc:
                if "capacity" in str(exc):
                    event = ProtocolSession(sample_rate=16000).error(exc, code="overloaded")
                    await websocket.send_json(event); terminal_sent = True
                    await websocket.close(code=1013); return
                raise
            except (ValueError, TimeoutError) as exc:
                event = ProtocolSession(sample_rate=16000).error(exc, code="invalid_start")
                await websocket.send_json(event); terminal_sent = True
                await websocket.close(code=1008); return

            await websocket.send_json(protocol.ready(session_id=session_id, worker_id=session.selected_worker_id))

            async def send_results() -> None:
                queue = runtime_value.event_queue(session_id)
                while True:
                    event = await queue.get()
                    await websocket.send_json(event)
                    if event["type"] in {"final", "error"}:
                        return

            sender = asyncio.create_task(send_results())
            while True:
                incoming = await asyncio.wait_for(websocket.receive(), timeout=runtime_value.settings.asr_idle_timeout_seconds)
                if incoming["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(incoming.get("code", 1000))
                if incoming.get("bytes") is not None:
                    try:
                        await runtime_value.ingest(session, incoming["bytes"])
                    except (ValueError, BufferError) as exc:
                        await websocket.send_json(protocol.error(exc, code="invalid_audio")); terminal_sent = True
                        await websocket.close(code=1008); return
                    continue
                try:
                    control = parse_client_command(json.loads(incoming.get("text") or "{}"))
                except (ValueError, json.JSONDecodeError) as exc:
                    await websocket.send_json(protocol.error(exc, code="invalid_command")); terminal_sent = True
                    await websocket.close(code=1008); return
                if control.type == "segment":
                    for event in await runtime_value.segment(session_id):
                        await websocket.send_json(event)
                elif control.type == "abort":
                    await runtime_value.abort(session_id)
                    await websocket.close(code=1000); return
                elif control.type == "finish":
                    event = await runtime_value.finish(session_id)
                    await websocket.send_json(event); terminal_sent = True
                    await websocket.close(code=1000); return
                else:
                    raise ValueError("start may only be sent once")
        except (WebSocketDisconnect, TimeoutError):
            return
        except Exception as exc:
            if protocol is not None and not protocol.terminal and not terminal_sent:
                with suppress(Exception):
                    await websocket.send_json(protocol.error(exc))
        finally:
            if sender is not None:
                sender.cancel()
                with suppress(asyncio.CancelledError):
                    await sender
            await runtime_value.abort(session_id)

    return app


app = create_app()
