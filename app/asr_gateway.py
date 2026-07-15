from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Mapping

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.asr_gateway_backends import BackendLease, BackendRegistry, ResultMode, VadMode
from app.asr_gateway_metrics import GatewayMetrics, JobTimeline, STAGES, gateway_readiness
from app.asr_gateway_protocol import ProtocolSession, StartCommand, parse_client_command
from app.asr_gateway_scheduler import BatchKey, GatewayScheduler, InferenceJob, InferenceResult, StaleResultError
from app.asr_gateway_sessions import GatewaySession, SessionManager, TerminalState
from app.config import Settings, get_settings


@dataclass
class _SessionContext:
    session: GatewaySession
    lease: BackendLease
    protocol: ProtocolSession
    events: asyncio.Queue[dict[str, Any]]
    idle: asyncio.Event
    vad: Any | None = None
    endpoint_pending: bool = False
    utterance_samples: int = 0
    opened_at: float = 0.0
    first_undecoded_at: float | None = None


class GatewayRuntime:
    def __init__(
        self,
        settings: Settings,
        adapters: Mapping[str, Any],
        *,
        vad_factory: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        self.settings = settings
        self.adapters = dict(adapters)
        self.registry = BackendRegistry()
        self.sessions = SessionManager(max_sessions=settings.asr_gateway_max_active_sessions)
        self.metrics = GatewayMetrics()
        self._contexts: dict[str, _SessionContext] = {}
        self._vad_factory = vad_factory
        self._timelines: dict[str, tuple[JobTimeline, int, int]] = {}
        self._worker_empty: dict[str, asyncio.Event] = {
            worker_id: asyncio.Event() for worker_id in self.adapters
        }
        for event in self._worker_empty.values():
            event.set()
        self.scheduler = GatewayScheduler(
            self.adapters,
            clock=clock or time.monotonic,
            max_wait_seconds=settings.asr_gateway_schedule_max_wait_ms / 1000,
            max_ready_jobs=settings.asr_gateway_max_ready_jobs,
            max_queued_samples=round(settings.asr_gateway_max_queued_audio_seconds * 16_000),
            cleanup=self._cleanup_job,
            publish=self._publish_result,
            worker_failed=self._worker_failed,
            stage_hook=self._record_stage,
            reject=self._reject_job,
            inference_timeout_seconds=settings.asr_stream_inference_timeout_seconds,
        )
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        try:
            # Validate immutable ownership and topology before any model warmup.
            for worker_id, adapter in self.adapters.items():
                if adapter.capabilities.worker_id != worker_id:
                    raise ValueError("adapter map key must equal capability worker_id")
                await self.registry.register(adapter.capabilities)
            for worker_id, adapter in self.adapters.items():
                await adapter.warmup()
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
        task = str(options.get("task", "transcribe"))
        lease = await self.registry.acquire(
            backend_id=self.settings.asr_gateway_default_backend,
            language=language,
            task=task,
            streaming_mode=self._required_streaming_mode(),
            result_modes=(
                ResultMode.CUMULATIVE_SNAPSHOT,
                ResultMode.REPLACEABLE_SEGMENT,
                ResultMode.CONFIRMED_PLUS_TAIL,
            ),
            model_id=None if self.settings.asr_backend == "mock" else self.settings.asr_model_id,
        )
        worker_id = lease.worker_id
        self._worker_empty[worker_id].clear()
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
            self._worker_empty[worker_id].set()
            raise
        idle = asyncio.Event(); idle.set()
        vad = None
        if adapter.capabilities.vad_mode is VadMode.GATEWAY:
            if self._vad_factory is not None:
                vad = self._vad_factory()
            else:
                from app.asr_vad import create_vad_endpoint_detector
                vad = create_vad_endpoint_detector(self.settings)
        self._contexts[session_id] = _SessionContext(
            session, lease,
            ProtocolSession(
                sample_rate=16_000,
                segment_local=adapter.capabilities.result_mode is ResultMode.REPLACEABLE_SEGMENT,
            ),
            asyncio.Queue(), idle, vad, False, 0, self.scheduler.clock(), None,
        )
        return session

    def _required_streaming_mode(self):
        from app.asr_gateway_backends import StreamingMode
        return StreamingMode.STATEFUL if self.settings.asr_stream_mode == "stateful" else StreamingMode.CHUNKED

    async def ingest(self, session: GatewaySession, pcm: bytes, *, force: bool = False) -> bool:
        ctx = self._contexts[session.session_id]
        self.check_deadlines(session.session_id)
        incoming_samples = len(pcm) // 2
        if session.sample_accounting["accepted"] + incoming_samples > round(
            self.settings.asr_max_audio_seconds * session.sample_rate
        ):
            raise RuntimeError("audio duration limit exceeded")
        if pcm and ctx.first_undecoded_at is None:
            ctx.first_undecoded_at = self.scheduler.clock()
        if pcm and ctx.vad is not None:
            session.accept_vad_input(pcm)
            decision = ctx.vad.add_audio(pcm)
            if decision.audio_to_model:
                session.append_pcm(decision.audio_to_model, count_accepted=False)
            if decision.discarded_samples:
                session.record_discarded(decision.discarded_samples)
            ctx.endpoint_pending = ctx.endpoint_pending or decision.endpoint
            force = force or decision.endpoint
        elif pcm:
            session.append_pcm(pcm)
        return self._schedule_next(session, force=force)

    def check_deadlines(self, session_id: str) -> None:
        ctx = self._contexts[session_id]
        now = self.scheduler.clock()
        if now - ctx.opened_at > self.settings.asr_max_session_seconds:
            raise TimeoutError("session deadline exceeded")
        if (
            ctx.first_undecoded_at is not None
            and now - ctx.first_undecoded_at > self.settings.asr_max_undecoded_age_seconds
        ):
            raise TimeoutError("undecoded audio age exceeded")

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
            queue_deadline=self.scheduler.clock() + self.settings.asr_stream_queue_timeout_seconds,
        )
        try:
            now = self.scheduler.clock()
            timeline = JobTimeline(job.job_id, job.worker_id, job.sample_count)
            timeline.mark("audio_received", now)
            timeline.mark("chunk_ready", now)
            timeline.mark("scheduler_enqueued", now)
            self._timelines[job.job_id] = (timeline, 1, caps.max_batch_items)
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
            self._apply_control_result(ctx, result.text)
        ctx.utterance_samples = 0
        return ctx.protocol.segment()

    async def finish(self, session_id: str) -> dict[str, Any]:
        ctx = self._contexts[session_id]
        if ctx.vad is not None:
            decision = ctx.vad.finish_input()
            if decision.audio_to_model:
                ctx.session.append_pcm(decision.audio_to_model, count_accepted=False)
            if decision.discarded_samples:
                ctx.session.record_discarded(decision.discarded_samples)
        ctx.session.request_finish()
        await self.ingest(ctx.session, b"", force=True)
        await ctx.idle.wait()
        result = await self.adapters[ctx.session.selected_worker_id].finish_session(session_id)
        if result.text:
            self._apply_control_result(ctx, result.text)
        event = ctx.protocol.final()
        ctx.session.succeed()
        await self._release_session(session_id)
        return event

    async def abort(self, session_id: str) -> None:
        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        self.scheduler.cancel_session(session_id, generation=ctx.session.generation)
        await self.scheduler.wait_session_safe(session_id, generation=ctx.session.generation)
        self.metrics.cancellations += 1
        ctx.session.abort()
        if ctx.vad is not None:
            ctx.vad.reset()
        with suppress(Exception):
            await self.adapters[ctx.session.selected_worker_id].abort_session(session_id)
        await self._release_session(session_id)

    async def _release_session(self, session_id: str) -> None:
        ctx = self._contexts.pop(session_id, None)
        self.sessions.close(session_id)
        if ctx is not None:
            await ctx.lease.release()
            if not any(
                item.session.selected_worker_id == ctx.session.selected_worker_id
                for item in self._contexts.values()
            ):
                self._worker_empty[ctx.session.selected_worker_id].set()

    async def drain_worker(self, worker_id: str) -> None:
        await self.registry.begin_drain(worker_id)
        try:
            await asyncio.wait_for(
                self._worker_empty[worker_id].wait(),
                timeout=self.settings.asr_gateway_drain_timeout_seconds,
            )
        except TimeoutError:
            for session_id, ctx in list(self._contexts.items()):
                if ctx.session.selected_worker_id == worker_id:
                    await self.abort(session_id)
            await self._worker_empty[worker_id].wait()
        adapter = self.adapters[worker_id]
        if hasattr(adapter, "drain"):
            await adapter.drain()
        await self.registry.remove(worker_id)

    async def _cleanup_job(self, job: InferenceJob) -> None:
        ctx = self._contexts.get(job.session_id)
        if (
            ctx is None
            or ctx.session.generation != job.generation
            or not ctx.session.matches_reservation(
                job.job_sequence, job.start_sample, job.end_sample
            )
        ):
            self.metrics.conflicts += 1
            raise StaleResultError("job no longer owns the session reservation")
        ctx.session.acknowledge(job.job_sequence, generation=job.generation)
        self._update_gauges()

    async def _reject_job(self, job: InferenceJob) -> None:
        ctx = self._contexts.get(job.session_id)
        if ctx is None or ctx.session.generation != job.generation:
            return
        if not ctx.session.rollback(job.job_sequence):
            self.metrics.conflicts += 1
            raise StaleResultError("rejected job lost its reservation")

    async def _worker_failed(self, worker_id: str, reason: str) -> None:
        self.metrics.failures += 1
        await self.registry.mark_ready(worker_id, False, error=reason)

    def _record_stage(self, stage: str, jobs: Any, capacity: int) -> None:
        now = self.scheduler.clock()
        for job in jobs:
            record = self._timelines.get(job.job_id)
            if record is None:
                continue
            timeline, _, _ = record
            timeline.mark(stage, now)
            self._timelines[job.job_id] = (timeline, len(jobs), capacity)

    def _complete_timeline(self, job_id: str) -> None:
        record = self._timelines.pop(job_id, None)
        if record is None:
            return
        timeline, batch_size, capacity = record
        now = self.scheduler.clock()
        for stage in STAGES:
            if stage not in timeline.stages:
                timeline.mark(stage, now)
        self.metrics.complete(timeline, batch_size=batch_size, batch_capacity=capacity)

    async def _publish_result(self, result: InferenceResult) -> None:
        try:
            await self._apply_result(result)
        except Exception as exc:
            ctx = self._contexts.get(result.session_id)
            if ctx is None:
                return
            self.metrics.conflicts += 1
            ctx.session.fail()
            if not ctx.protocol.terminal:
                await ctx.events.put(ctx.protocol.error(exc, code="result_conflict"))
            ctx.idle.set()
            self._complete_timeline(result.job_id)

    async def _apply_result(self, result: InferenceResult) -> None:
        ctx = self._contexts.get(result.session_id)
        if ctx is None:
            return
        now = self.scheduler.clock()
        record = self._timelines.get(result.job_id)
        received_at = record[0].stages.get("audio_received", now) if record else now
        if now - received_at > self.settings.asr_max_connection_lag_seconds:
            self.metrics.failures += 1
            ctx.session.fail()
            await ctx.events.put(
                ctx.protocol.error(TimeoutError("connection lag exceeded"), code="audio_lag")
            )
            ctx.idle.set()
            self._complete_timeline(result.job_id)
            return
        if (
            ctx.first_undecoded_at is not None
            and now - ctx.first_undecoded_at > self.settings.asr_max_undecoded_age_seconds
        ):
            self.metrics.failures += 1
            ctx.session.fail()
            await ctx.events.put(
                ctx.protocol.error(TimeoutError("undecoded audio age exceeded"), code="audio_lag")
            )
            ctx.idle.set()
            self._complete_timeline(result.job_id)
            return
        if result.error:
            ctx.session.fail()
            await ctx.events.put(ctx.protocol.error(RuntimeError(result.error)))
            ctx.idle.set()
            self._complete_timeline(result.job_id)
            return
        mode = self.adapters[result.worker_id].capabilities.result_mode
        text = result.text
        if (
            mode is ResultMode.CUMULATIVE_SNAPSHOT
            and ctx.protocol.state.confirmed_text
            and not text.startswith(ctx.protocol.state.confirmed_text)
        ):
            text = ctx.protocol.state.confirmed_text + text
        events = ctx.protocol.apply_result(
            mode, text=text, confirmed_text=result.confirmed_text,
            tail_text=result.tail_text, decoded_samples=result.end_sample - result.start_sample,
            segment_id=result.job_sequence,
        )
        for event in events:
            await ctx.events.put(event)
        ctx.utterance_samples += result.end_sample - result.start_sample
        caps = self.adapters[result.worker_id].capabilities
        continuation_ready = False
        if ctx.endpoint_pending or ctx.utterance_samples >= caps.max_input_samples:
            control = await self.adapters[result.worker_id].finish_segment(result.session_id)
            if control.text:
                self._apply_control_result(ctx, control.text)
            for event in ctx.protocol.segment():
                await ctx.events.put(event)
            ctx.utterance_samples = 0
            if ctx.endpoint_pending and ctx.vad is not None:
                ctx.endpoint_pending = False
                decision = ctx.vad.endpoint_finalized()
                if decision.audio_to_model:
                    ctx.session.append_pcm(decision.audio_to_model, count_accepted=False)
                    continuation_ready = True
                if decision.discarded_samples:
                    ctx.session.record_discarded(decision.discarded_samples)
                ctx.endpoint_pending = decision.endpoint
        scheduled = self._schedule_next(
            ctx.session,
            force=continuation_ready or ctx.endpoint_pending or ctx.session.finish_requested,
        )
        if (
            not scheduled
            and not ctx.session.in_flight
            and ctx.session.buffer.buffered_samples == 0
            and ctx.session.sample_accounting["pending_vad"] == 0
        ):
            ctx.idle.set()
            ctx.first_undecoded_at = None
        self._complete_timeline(result.job_id)

    def _apply_control_result(self, ctx: _SessionContext, text: str) -> None:
        mode = self.adapters[ctx.session.selected_worker_id].capabilities.result_mode
        if mode is ResultMode.REPLACEABLE_SEGMENT:
            segment_id = ctx.protocol.state.active_segment_id or 0
            ctx.protocol.apply_result(mode, text=text, segment_id=segment_id)
        else:
            ctx.protocol.apply_result(
                ResultMode.CUMULATIVE_SNAPSHOT,
                text=ctx.protocol.state.confirmed_text + text,
            )

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
                try:
                    incoming = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=runtime_value.settings.asr_idle_timeout_seconds,
                    )
                except TimeoutError as exc:
                    await websocket.send_json(protocol.error(exc, code="idle_timeout"))
                    terminal_sent = True
                    await websocket.close(code=1011)
                    return
                try:
                    runtime_value.check_deadlines(session_id)
                except TimeoutError as exc:
                    await websocket.send_json(protocol.error(exc, code="session_timeout"))
                    terminal_sent = True
                    await websocket.close(code=1011)
                    return
                if incoming["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(incoming.get("code", 1000))
                if incoming.get("bytes") is not None:
                    try:
                        await runtime_value.ingest(session, incoming["bytes"])
                    except (ValueError, BufferError) as exc:
                        await websocket.send_json(protocol.error(exc, code="invalid_audio")); terminal_sent = True
                        await websocket.close(code=1008); return
                    except TimeoutError as exc:
                        await websocket.send_json(protocol.error(exc, code="audio_lag")); terminal_sent = True
                        await websocket.close(code=1011); return
                    except RuntimeError as exc:
                        await websocket.send_json(protocol.error(exc, code="audio_limit")); terminal_sent = True
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
