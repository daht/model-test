from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.asr_gateway_backends import BackendLease, BackendRegistry, ResultMode, VadMode
from app.asr_gateway_metrics import GatewayMetrics, JobTimeline, gateway_readiness
from app.asr_observability import (
    CapacityBufferError,
    configure_events,
    events as observability_events,
)
from app.asr_gateway_protocol import ProtocolSession, StartCommand, parse_client_command
from app.asr_gateway_scheduler import BatchKey, GatewayScheduler, InferenceJob, InferenceResult, StaleResultError
from app.asr_gateway_sessions import GatewaySession, SessionManager, TerminalState
from app.config import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutboundEvent:
    payload: dict[str, Any] | None
    job_id: str | None = None
    completes_timeline: bool = False
    terminal: bool = False
    close_code: int | None = None


@dataclass
class _SessionContext:
    session: GatewaySession
    lease: BackendLease
    protocol: ProtocolSession
    events: asyncio.Queue[OutboundEvent]
    idle: asyncio.Event
    vad: Any | None = None
    endpoint_pending: bool = False
    utterance_samples: int = 0
    opened_at: float = 0.0
    first_undecoded_at: float | None = None
    protocol_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lease_released: bool = False
    segment_metadata: dict[str, Any] | None = None


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
        configure_events(
            diagnostic_enabled=settings.asr_diagnostic_logging,
            slow_engine_seconds=settings.asr_slow_engine_log_seconds,
        )
        self.adapters = dict(adapters)
        self.registry = BackendRegistry()
        self.sessions = SessionManager(max_sessions=settings.asr_gateway_max_active_sessions)
        self.metrics = GatewayMetrics()
        for adapter in self.adapters.values():
            if hasattr(adapter, "set_engine_observer"):
                adapter.set_engine_observer(self.metrics.record_engine_call)
            if hasattr(adapter, "set_capacity_observer"):
                adapter.set_capacity_observer(self.metrics.record_capacity_rejection)
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
            discard=self._discard_job,
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
            observability_events().emit(
                "asr_process_started",
                component="gateway",
                backend=self.settings.asr_backend,
                stream_mode=self.settings.asr_stream_mode,
                diagnostic_enabled=self.settings.asr_diagnostic_logging,
                worker_count=len(self.adapters),
            )
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
        self._update_gauges()
        observability_events().emit(
            "asr_session_opened",
            component="gateway",
            session_id=session_id,
            generation=session.generation,
            worker_id=worker_id,
            language=language,
            active_sessions=self.sessions.snapshot()["active_sessions"],
        )
        return session

    def _required_streaming_mode(self):
        from app.asr_gateway_backends import StreamingMode
        return {
            "stateful": StreamingMode.STATEFUL,
            "chunked": StreamingMode.CHUNKED,
            "rolling": StreamingMode.ROLLING,
        }[self.settings.asr_stream_mode]

    async def ingest(self, session: GatewaySession, pcm: bytes, *, force: bool = False) -> bool:
        ctx = self._contexts[session.session_id]
        async with ctx.protocol_lock:
            return await self._ingest_locked(ctx, session, pcm, force=force)

    async def _ingest_locked(
        self,
        ctx: _SessionContext,
        session: GatewaySession,
        pcm: bytes,
        *,
        force: bool,
    ) -> bool:
        self.check_deadlines(session.session_id)
        incoming_samples = len(pcm) // 2
        if session.sample_accounting["accepted"] + incoming_samples > round(
            self.settings.asr_max_audio_seconds * session.sample_rate
        ):
            raise RuntimeError("audio duration limit exceeded")
        if pcm and ctx.vad is not None:
            await self._accept_vad_with_backpressure(session, pcm)
            decision = ctx.vad.add_audio(pcm)
            if decision.audio_to_model:
                await self._append_pcm_with_backpressure(
                    session, decision.audio_to_model, count_accepted=False
                )
            if decision.discarded_samples:
                session.record_discarded(decision.discarded_samples)
            ctx.endpoint_pending = ctx.endpoint_pending or decision.endpoint
            force = force or decision.endpoint
        elif pcm:
            await self._append_pcm_with_backpressure(session, pcm)
        scheduled = self._schedule_next(session, force=force)
        self._update_gauges()
        accounting = session.sample_accounting
        observability_events().emit(
            "asr_audio_ingested",
            component="gateway",
            diagnostic=True,
            session_id=session.session_id,
            generation=session.generation,
            incoming_samples=incoming_samples,
            buffered_samples=accounting["buffered"],
            reserved_samples=accounting["reserved"],
            pending_vad_samples=accounting["pending_vad"],
            scheduled=scheduled,
        )
        return scheduled

    async def _accept_vad_with_backpressure(
        self,
        session: GatewaySession,
        pcm: bytes,
    ) -> None:
        while True:
            try:
                session.accept_vad_input(pcm)
                return
            except CapacityBufferError as exc:
                if exc.reason != "session_pcm_limit" or not session.in_flight:
                    raise
                await asyncio.wait_for(
                    session.wait_reservation_released(),
                    timeout=self.settings.asr_stream_queue_timeout_seconds,
                )
                self.check_deadlines(session.session_id)

    async def _append_pcm_with_backpressure(
        self,
        session: GatewaySession,
        pcm: bytes,
        *,
        count_accepted: bool = True,
    ) -> None:
        while True:
            try:
                session.append_pcm(pcm, count_accepted=count_accepted)
                return
            except CapacityBufferError as exc:
                if exc.reason != "session_pcm_limit" or not session.in_flight:
                    raise
                await asyncio.wait_for(
                    session.wait_reservation_released(),
                    timeout=self.settings.asr_stream_queue_timeout_seconds,
                )
                self.check_deadlines(session.session_id)

    def check_deadlines(self, session_id: str) -> None:
        ctx = self._contexts[session_id]
        now = self.scheduler.clock()
        if now - ctx.opened_at > self.settings.asr_max_session_seconds:
            raise TimeoutError("session deadline exceeded")
        preferred = round(
            self.settings.asr_gateway_default_update_ms
            / 1000
            * ctx.session.sample_rate
        )
        schedulable = bool(
            ctx.session.in_flight
            or ctx.session.buffer.buffered_samples >= preferred
            or ctx.endpoint_pending
            or ctx.session.finish_requested
        )
        if (
            ctx.first_undecoded_at is not None
            and schedulable
            and now - ctx.first_undecoded_at > self.settings.asr_max_undecoded_age_seconds
        ):
            raise TimeoutError("undecoded audio age exceeded")

    def _schedule_next(self, session: GatewaySession, *, force: bool = False) -> bool:
        preferred = round(self.settings.asr_gateway_default_update_ms / 1000 * session.sample_rate)
        adapter = self.adapters[session.selected_worker_id]
        caps = adapter.capabilities
        ctx = self._contexts[session.session_id]
        segment_remaining = caps.max_segment_samples - ctx.utterance_samples
        adapter_remaining = getattr(adapter, "remaining_segment_samples", None)
        if callable(adapter_remaining):
            segment_remaining = min(
                segment_remaining,
                adapter_remaining(session.session_id),
            )
        maximum = min(caps.max_input_samples, segment_remaining)
        count = session.ready_samples(
            preferred=preferred,
            maximum=maximum,
            force=force,
        )
        if count <= 0:
            return False
        final_decode = bool(
            session.finish_requested
            or ctx.endpoint_pending
            or count == segment_remaining
        )
        reservation = session.reserve(count, final=final_decode)
        ctx.idle.clear()
        options = session.options
        options_identity = str(hash(json.dumps(options, sort_keys=True, default=str)))
        rolling_batch = self.settings.asr_backend in {"faster_whisper", "sensevoice"}
        bucket = 0 if rolling_batch else min(15, max(0, count.bit_length() - 1))
        decoding_identity = (
            ("final:" if reservation.chunk.final else "partial:")
            + options_identity
        )
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
                decoding_identity,
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
            if ctx.first_undecoded_at is None:
                ctx.first_undecoded_at = now
            observability_events().emit(
                "asr_job_enqueued",
                component="gateway",
                diagnostic=True,
                session_id=job.session_id,
                generation=job.generation,
                job_id=job.job_id,
                worker_id=job.worker_id,
                chunk_samples=job.sample_count,
                final=job.final,
                length_bucket=job.batch_key.length_bucket,
                ready_depth=self.scheduler.snapshot()["ready_depth"],
            )
        except Exception:
            self.discard_timeline(job.job_id)
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
        async with ctx.protocol_lock:
            result = await self.adapters[ctx.session.selected_worker_id].finish_segment(session_id)
            metadata = getattr(result, "metadata", None)
            events = self._apply_control_result(ctx, result.text, metadata) if result.text else []
            if metadata is not None:
                ctx.segment_metadata = dict(metadata)
            events.extend(ctx.protocol.segment(metadata=ctx.segment_metadata))
            ctx.segment_metadata = None
            ctx.utterance_samples = 0
            await self._enqueue_events(ctx, events)
            return events

    async def enqueue_segment(self, session_id: str) -> None:
        await self.segment(session_id)

    async def finish(self, session_id: str) -> dict[str, Any]:
        ctx = self._contexts[session_id]
        if ctx.vad is not None:
            decision = ctx.vad.finish_input()
            if decision.audio_to_model:
                ctx.session.append_pcm(decision.audio_to_model, count_accepted=False)
            if decision.discarded_samples:
                ctx.session.record_discarded(decision.discarded_samples)
        ctx.session.request_finish()
        scheduled = await self.ingest(ctx.session, b"", force=True)
        if (
            not scheduled
            and not ctx.session.in_flight
            and ctx.session.buffer.buffered_samples == 0
            and ctx.session.sample_accounting["pending_vad"] == 0
        ):
            ctx.idle.set()
            ctx.first_undecoded_at = None
        await ctx.idle.wait()
        async with ctx.protocol_lock:
            result = await self.adapters[ctx.session.selected_worker_id].finish_session(session_id)
            metadata = getattr(result, "metadata", None)
            events = self._apply_control_result(ctx, result.text, metadata) if result.text else []
            if metadata is not None:
                ctx.segment_metadata = dict(metadata)
            event = ctx.protocol.final(metadata=ctx.segment_metadata)
            ctx.segment_metadata = None
            events.append(event)
            ctx.session.succeed()
            observability_events().emit(
                "asr_session_terminal",
                component="gateway",
                session_id=session_id,
                generation=ctx.session.generation,
                terminal_state="succeeded",
                reason="finished",
                close_code=1000,
            )
            await ctx.lease.release()
            ctx.lease_released = True
            await self._enqueue_events(
                ctx,
                events,
                terminal=True,
                close_code=1000,
            )
            return event

    async def enqueue_finish(self, session_id: str) -> None:
        await self.finish(session_id)

    async def enqueue_ready(self, session_id: str) -> None:
        ctx = self._contexts[session_id]
        async with ctx.protocol_lock:
            await self._enqueue_events(
                ctx,
                [
                    ctx.protocol.ready(
                        session_id=session_id,
                        worker_id=ctx.session.selected_worker_id,
                    )
                ],
            )

    async def enqueue_error(
        self,
        session_id: str,
        exc: BaseException,
        *,
        code: str = "backend_error",
        close_code: int = 1011,
    ) -> None:
        ctx = self._contexts[session_id]
        generation = ctx.session.generation
        self.scheduler.cancel_session(session_id, generation=generation)
        await self.scheduler.wait_session_safe(session_id, generation=generation)
        async with ctx.protocol_lock:
            if ctx.protocol.terminal:
                return
            self.metrics.failures += 1
            logger.warning(
                "asr_gateway_session_error session_id=%s code=%s exception_type=%s",
                session_id,
                code,
                type(exc).__name__,
            )
            if isinstance(exc, CapacityBufferError):
                self.metrics.record_capacity_rejection(exc.reason)
                observability_events().emit(
                    "asr_buffer_rejected",
                    component="gateway",
                    session_id=session_id,
                    generation=generation,
                    reason=exc.reason,
                    **exc.safe_fields,
                )
            await self._fail_session(
                ctx,
                reason=(exc.reason if isinstance(exc, CapacityBufferError) else code),
                close_code=close_code,
                exception_type=type(exc).__name__,
            )
            await self._enqueue_events(
                ctx,
                [ctx.protocol.error(exc, code=code)],
                terminal=True,
                close_code=close_code,
            )

    async def enqueue_close(self, session_id: str, *, close_code: int = 1000) -> None:
        ctx = self._contexts[session_id]
        await ctx.events.put(
            OutboundEvent(None, terminal=True, close_code=close_code)
        )

    async def enqueue_abort(self, session_id: str) -> None:
        ctx = self._contexts[session_id]
        self.scheduler.cancel_session(session_id, generation=ctx.session.generation)
        await self.scheduler.wait_session_safe(
            session_id, generation=ctx.session.generation
        )
        self.metrics.cancellations += 1
        ctx.session.abort()
        observability_events().emit(
            "asr_session_terminal",
            component="gateway",
            session_id=session_id,
            generation=ctx.session.generation,
            terminal_state="aborted",
            reason="client_abort",
            close_code=1000,
        )
        if ctx.vad is not None:
            ctx.vad.reset()
        await self.adapters[ctx.session.selected_worker_id].abort_session(session_id)
        await self.enqueue_close(session_id, close_code=1000)

    async def abort(self, session_id: str) -> None:
        ctx = self._contexts.get(session_id)
        if ctx is None:
            return
        self.scheduler.cancel_session(session_id, generation=ctx.session.generation)
        await self.scheduler.wait_session_safe(session_id, generation=ctx.session.generation)
        self.metrics.cancellations += 1
        ctx.session.abort()
        observability_events().emit(
            "asr_session_terminal",
            component="gateway",
            session_id=session_id,
            generation=ctx.session.generation,
            terminal_state="aborted",
            reason="connection_abort",
        )
        if ctx.vad is not None:
            ctx.vad.reset()
        with suppress(Exception):
            await self.adapters[ctx.session.selected_worker_id].abort_session(session_id)
        await self._release_session(session_id)

    async def _fail_session(
        self,
        ctx: _SessionContext,
        *,
        reason: str,
        close_code: int = 1011,
        exception_type: str | None = None,
    ) -> None:
        ctx.session.fail()
        with suppress(Exception):
            await self.adapters[ctx.session.selected_worker_id].abort_session(
                ctx.session.session_id
            )
        observability_events().emit(
            "asr_session_terminal",
            component="gateway",
            session_id=ctx.session.session_id,
            generation=ctx.session.generation,
            terminal_state="failed",
            reason=reason,
            close_code=close_code,
            exception_type=exception_type,
        )

    async def _release_session(self, session_id: str) -> None:
        ctx = self._contexts.pop(session_id, None)
        self.sessions.close(session_id)
        if ctx is not None:
            accounting = ctx.session.sample_accounting
            if not ctx.lease_released:
                await ctx.lease.release()
                ctx.lease_released = True
            if not any(
                item.session.selected_worker_id == ctx.session.selected_worker_id
                for item in self._contexts.values()
            ):
                self._worker_empty[ctx.session.selected_worker_id].set()
            observability_events().emit(
                "asr_session_released",
                component="gateway",
                session_id=session_id,
                generation=ctx.session.generation,
                buffered_samples=accounting["buffered"],
                reserved_samples=accounting["reserved"],
                pending_vad_samples=accounting["pending_vad"],
                lease_released=ctx.lease_released,
            )
        self._update_gauges()

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
            observability_events().emit(
                "asr_cleanup_conflict",
                component="gateway",
                session_id=job.session_id,
                generation=job.generation,
                job_id=job.job_id,
                current_generation=(ctx.session.generation if ctx is not None else None),
                reservation_present=bool(ctx and ctx.session.in_flight),
            )
            raise StaleResultError("job no longer owns the session reservation")
        ctx.session.acknowledge(job.job_sequence, generation=job.generation)
        ctx.first_undecoded_at = None
        self._update_gauges()
        accounting = ctx.session.sample_accounting
        observability_events().emit(
            "asr_job_cleaned",
            component="gateway",
            diagnostic=True,
            session_id=job.session_id,
            generation=job.generation,
            job_id=job.job_id,
            outcome="acknowledged",
            buffered_samples=accounting["buffered"],
            reserved_samples=accounting["reserved"],
        )

    async def _reject_job(self, job: InferenceJob) -> None:
        self.discard_timeline(job.job_id)
        ctx = self._contexts.get(job.session_id)
        if ctx is None or ctx.session.generation != job.generation:
            return
        if not ctx.session.rollback(job.job_sequence):
            self.metrics.conflicts += 1
            raise StaleResultError("rejected job lost its reservation")

    async def _discard_job(self, job: InferenceJob) -> None:
        self.discard_timeline(job.job_id)

    async def _worker_failed(self, worker_id: str, reason: str) -> None:
        self.metrics.failures += 1
        await self.registry.mark_ready(worker_id, False, error=reason)

    def _record_stage(self, stage: str, jobs: Any, capacity: int) -> None:
        now = self.scheduler.clock()
        if stage == "scheduler_dispatched":
            self.metrics.record_scheduler_batch(len(jobs))
        for job in jobs:
            record = self._timelines.get(job.job_id)
            if record is None:
                continue
            timeline, _, _ = record
            timeline.mark(stage, now)
            self._timelines[job.job_id] = (timeline, len(jobs), capacity)

    def _mark_result_applied(self, job_id: str) -> None:
        record = self._timelines.get(job_id)
        if record is None:
            return
        timeline, _, _ = record
        now = self.scheduler.clock()
        if "result_applied" not in timeline.stages:
            timeline.mark("result_applied", now)

    def mark_event_sent(self, job_id: str) -> None:
        record = self._timelines.pop(job_id, None)
        if record is None:
            return
        timeline, batch_size, capacity = record
        timeline.mark("event_sent", self.scheduler.clock())
        self.metrics.complete(timeline, batch_size=batch_size, batch_capacity=capacity)

    def discard_timeline(self, job_id: str) -> None:
        self._timelines.pop(job_id, None)

    async def _enqueue_events(
        self,
        ctx: _SessionContext,
        events: list[dict[str, Any]],
        *,
        job_id: str | None = None,
        terminal: bool = False,
        close_code: int | None = None,
    ) -> None:
        for index, event in enumerate(events):
            last = index == len(events) - 1
            await ctx.events.put(
                OutboundEvent(
                    event,
                    job_id=job_id,
                    completes_timeline=bool(job_id and last),
                    terminal=terminal and last,
                    close_code=close_code if terminal and last else None,
                )
            )

    async def _publish_result(self, result: InferenceResult) -> None:
        try:
            await self._apply_result(result)
        except Exception as exc:
            ctx = self._contexts.get(result.session_id)
            if ctx is None:
                return
            capacity_failure = isinstance(exc, CapacityBufferError)
            if capacity_failure:
                self.metrics.failures += 1
                self.metrics.record_capacity_rejection(exc.reason)
                observability_events().emit(
                    "asr_buffer_rejected",
                    component="gateway",
                    session_id=result.session_id,
                    generation=result.generation,
                    job_id=result.job_id,
                    reason=exc.reason,
                    **exc.safe_fields,
                )
            else:
                self.metrics.conflicts += 1
            async with ctx.protocol_lock:
                reason = exc.reason if capacity_failure else "result_conflict"
                code = "overloaded" if capacity_failure else "result_conflict"
                close_code = 1013 if capacity_failure else 1011
                await self._fail_session(
                    ctx,
                    reason=reason,
                    close_code=close_code,
                    exception_type=type(exc).__name__,
                )
                if not ctx.protocol.terminal:
                    event = ctx.protocol.error(exc, code=code)
                    self._mark_result_applied(result.job_id)
                    await self._enqueue_events(
                        ctx,
                        [event],
                        job_id=result.job_id,
                        terminal=True,
                        close_code=close_code,
                    )
                ctx.idle.set()

    async def _apply_result(self, result: InferenceResult) -> None:
        ctx = self._contexts.get(result.session_id)
        if ctx is None:
            return
        async with ctx.protocol_lock:
            state = deepcopy(ctx.protocol.state)
            terminal = ctx.protocol.terminal
            segment_metadata = deepcopy(ctx.segment_metadata)
            try:
                await self._apply_result_locked(ctx, result)
            except Exception:
                ctx.protocol.state = state
                ctx.protocol.terminal = terminal
                ctx.segment_metadata = segment_metadata
                raise

    async def _apply_result_locked(
        self,
        ctx: _SessionContext,
        result: InferenceResult,
    ) -> None:
        now = self.scheduler.clock()
        record = self._timelines.get(result.job_id)
        received_at = record[0].stages.get("audio_received", now) if record else now
        if now - received_at > self.settings.asr_max_connection_lag_seconds:
            self.metrics.failures += 1
            await self._fail_session(ctx, reason="connection_lag")
            event = ctx.protocol.error(
                TimeoutError("connection lag exceeded"), code="audio_lag"
            )
            self._mark_result_applied(result.job_id)
            await self._enqueue_events(
                ctx, [event], job_id=result.job_id, terminal=True, close_code=1011
            )
            ctx.idle.set()
            return
        if (
            ctx.first_undecoded_at is not None
            and now - ctx.first_undecoded_at > self.settings.asr_max_undecoded_age_seconds
        ):
            self.metrics.failures += 1
            await self._fail_session(ctx, reason="undecoded_audio_age")
            event = ctx.protocol.error(
                TimeoutError("undecoded audio age exceeded"), code="audio_lag"
            )
            self._mark_result_applied(result.job_id)
            await self._enqueue_events(
                ctx, [event], job_id=result.job_id, terminal=True, close_code=1011
            )
            ctx.idle.set()
            return
        if result.error:
            await self._fail_session(ctx, reason="adapter_result_error")
            event = ctx.protocol.error(RuntimeError(result.error))
            self._mark_result_applied(result.job_id)
            await self._enqueue_events(
                ctx, [event], job_id=result.job_id, terminal=True, close_code=1011
            )
            ctx.idle.set()
            return
        mode = self.adapters[result.worker_id].capabilities.result_mode
        if result.metadata is not None:
            ctx.segment_metadata = dict(result.metadata)
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
            segment_id=result.segment_id, metadata=ctx.segment_metadata,
        )
        ctx.utterance_samples += result.end_sample - result.start_sample
        caps = self.adapters[result.worker_id].capabilities
        deferred_endpoint_final = bool(
            ctx.endpoint_pending
            and not result.final
            and self._schedule_next(ctx.session, force=True)
        )
        continuation_ready = False
        if (
            result.final
            or (ctx.endpoint_pending and not deferred_endpoint_final)
            or ctx.utterance_samples >= caps.max_segment_samples
        ):
            control = await self.adapters[result.worker_id].finish_segment(result.session_id)
            control_metadata = getattr(control, "metadata", None)
            if control.text:
                events.extend(self._apply_control_result(ctx, control.text, control_metadata))
            elif control_metadata is not None:
                ctx.segment_metadata = dict(control_metadata)
            events.extend(ctx.protocol.segment(metadata=ctx.segment_metadata))
            ctx.segment_metadata = None
            ctx.utterance_samples = 0
            continuation_ready = ctx.session.buffer.buffered_samples > 0
            if ctx.endpoint_pending and ctx.vad is not None:
                ctx.endpoint_pending = False
                decision = ctx.vad.endpoint_finalized()
                if decision.audio_to_model:
                    ctx.session.append_pcm(decision.audio_to_model, count_accepted=False)
                    continuation_ready = True
                if decision.discarded_samples:
                    ctx.session.record_discarded(decision.discarded_samples)
                ctx.endpoint_pending = decision.endpoint
        scheduled = deferred_endpoint_final or self._schedule_next(
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
        self._mark_result_applied(result.job_id)
        if events:
            await self._enqueue_events(ctx, events, job_id=result.job_id)
        else:
            # No egress occurred, so this job is intentionally not completed.
            self.discard_timeline(result.job_id)
        observability_events().emit(
            "asr_result_published",
            component="gateway",
            diagnostic=True,
            session_id=result.session_id,
            generation=result.generation,
            job_id=result.job_id,
            worker_id=result.worker_id,
            final=result.final,
            emitted_events=len(events),
            scheduled_next=scheduled,
        )

    def _apply_control_result(
        self,
        ctx: _SessionContext,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if metadata is not None:
            ctx.segment_metadata = dict(metadata)
        mode = self.adapters[ctx.session.selected_worker_id].capabilities.result_mode
        if mode is ResultMode.REPLACEABLE_SEGMENT:
            segment_id = ctx.protocol.state.active_segment_id or 0
            return ctx.protocol.apply_result(
                mode, text=text, segment_id=segment_id, metadata=ctx.segment_metadata
            )
        return ctx.protocol.apply_result(
                ResultMode.CUMULATIVE_SNAPSHOT,
                text=ctx.protocol.state.confirmed_text + text,
                metadata=ctx.segment_metadata,
            )

    def event_queue(self, session_id: str) -> asyncio.Queue[OutboundEvent]:
        return self._contexts[session_id].events

    def _update_gauges(self) -> None:
        scheduler = self.scheduler.snapshot()
        accounting = [
            ctx.session.sample_accounting for ctx in self._contexts.values()
        ]
        buffered = sum(item["buffered"] for item in accounting)
        reserved = sum(item["reserved"] for item in accounting)
        max_held = max(
            (
                item["buffered"] + item["reserved"] + item["pending_vad"]
                for item in accounting
            ),
            default=0,
        )
        self.metrics.set_gauges(
            active_sessions=self.sessions.snapshot()["active_sessions"],
            ready_depth=scheduler["ready_depth"],
            queued_samples=scheduler["queued_samples"],
            session_buffered_samples=buffered,
            session_reserved_samples=reserved,
            max_session_held_samples=max_held,
            sample_rate=16_000,
        )


def _authorized(value: str | None, settings: Settings) -> bool:
    return bool(value and value == settings.api_key)


def _default_runtime() -> GatewayRuntime:
    settings = get_settings()
    if settings.asr_backend == "sensevoice":
        from app.asr_sensevoice import SenseVoiceAdapter, SenseVoiceEngine

        max_segment_samples = round(settings.asr_max_utterance_seconds * 16_000)
        adapter = SenseVoiceAdapter(
            lambda: SenseVoiceEngine(
                settings.asr_model_id,
                device=settings.asr_device,
                use_itn=settings.asr_sensevoice_use_itn,
            ),
            worker_id="local",
            model_id=settings.asr_model_id,
            model_revision=settings.asr_model_name,
            gpu_id=settings.asr_device,
            session_capacity=settings.asr_max_active_streams,
            batch_size=settings.asr_sensevoice_batch_size,
            max_segment_samples=max_segment_samples,
            model_manifest_path=settings.asr_model_manifest_path,
        )
        return GatewayRuntime(settings, {"local": adapter})
    if settings.asr_backend == "faster_whisper":
        from app.asr_faster_whisper import FasterWhisperAdapter, FasterWhisperEngine

        max_segment_samples = round(
            settings.asr_max_utterance_seconds * 16_000
        )
        adapter = FasterWhisperAdapter(
            lambda: FasterWhisperEngine(
                settings.asr_model_id,
                device=settings.asr_device,
                compute_type=settings.asr_faster_whisper_compute_type,
            ),
            worker_id="local",
            model_id=settings.asr_model_id,
            model_revision=settings.asr_model_name,
            gpu_id=settings.asr_device,
            session_capacity=settings.asr_max_active_streams,
            batch_size=settings.asr_faster_whisper_batch_size,
            partial_beam_size=settings.asr_faster_whisper_partial_beam_size,
            final_beam_size=settings.asr_faster_whisper_final_beam_size,
            max_segment_samples=max_segment_samples,
            model_manifest_path=settings.asr_model_manifest_path,
        )
        return GatewayRuntime(settings, {"local": adapter})

    from app.asr import create_asr_transcriber
    from app.asr_gateway_local_adapter import LocalCoordinatorAdapter
    from app.asr_inference import ASRInferenceCoordinator

    max_frame_samples = settings.asr_max_frame_bytes // 2
    adapter = LocalCoordinatorAdapter(
        lambda: ASRInferenceCoordinator(settings, lambda: create_asr_transcriber(settings)),
        worker_id="local", model_id=settings.asr_model_id,
        model_revision=settings.asr_model_name, gpu_id=settings.asr_device,
        session_capacity=settings.asr_max_active_streams,
        preferred_chunk_samples=min(
            round(settings.asr_stream_chunk_seconds * 16_000),
            max_frame_samples,
        ),
        max_input_samples=max_frame_samples,
        max_segment_samples=round(
            settings.asr_max_utterance_seconds * 16_000
        ),
    )
    return GatewayRuntime(settings, {"local": adapter})


async def send_outbound_events(
    runtime: GatewayRuntime,
    ctx: _SessionContext,
    websocket: WebSocket,
) -> None:
    while True:
        envelope = await ctx.events.get()
        try:
            if envelope.payload is not None:
                await websocket.send_json(envelope.payload)
            if envelope.completes_timeline and envelope.job_id is not None:
                runtime.mark_event_sent(envelope.job_id)
            if envelope.terminal:
                await websocket.close(code=envelope.close_code or 1000)
        finally:
            if envelope.terminal:
                if envelope.job_id is not None:
                    runtime.discard_timeline(envelope.job_id)
                await runtime._release_session(ctx.session.session_id)
        if envelope.terminal:
            return


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
        return {
            **current().metrics.snapshot(),
            "scheduler": current().scheduler.snapshot(),
        }

    @app.websocket("/v1/transcribe/stream")
    async def stream(websocket: WebSocket) -> None:
        runtime_value = current()
        if not _authorized(websocket.headers.get("x-api-key"), runtime_value.settings):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session_id = uuid.uuid4().hex
        protocol: ProtocolSession | None = None
        ctx: _SessionContext | None = None
        sender: asyncio.Task[None] | None = None
        try:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=runtime_value.settings.asr_start_timeout_seconds)
                command = parse_client_command(message)
                if not isinstance(command, StartCommand):
                    raise ValueError("first command must be start")
                session = await runtime_value.open_session(session_id, language=command.language, options=command.options)
                ctx = runtime_value._contexts[session_id]
                protocol = ctx.protocol
            except RuntimeError as exc:
                if "capacity" in str(exc):
                    event = ProtocolSession(sample_rate=16000).error(exc, code="overloaded")
                    await websocket.send_json(event)
                    await websocket.close(code=1013); return
                raise
            except (ValueError, TimeoutError) as exc:
                event = ProtocolSession(sample_rate=16000).error(exc, code="invalid_start")
                await websocket.send_json(event)
                await websocket.close(code=1008); return

            sender = asyncio.create_task(
                send_outbound_events(runtime_value, ctx, websocket)
            )
            await runtime_value.enqueue_ready(session_id)
            while True:
                try:
                    incoming = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=runtime_value.settings.asr_idle_timeout_seconds,
                    )
                except TimeoutError as exc:
                    await runtime_value.enqueue_error(
                        session_id, exc, code="idle_timeout", close_code=1011
                    )
                    await sender
                    return
                try:
                    runtime_value.check_deadlines(session_id)
                except TimeoutError as exc:
                    await runtime_value.enqueue_error(
                        session_id, exc, code="session_timeout", close_code=1011
                    )
                    await sender
                    return
                if incoming["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(incoming.get("code", 1000))
                if incoming.get("bytes") is not None:
                    try:
                        await runtime_value.ingest(session, incoming["bytes"])
                    except (ValueError, BufferError) as exc:
                        await runtime_value.enqueue_error(
                            session_id, exc, code="invalid_audio", close_code=1008
                        )
                        await sender; return
                    except TimeoutError as exc:
                        await runtime_value.enqueue_error(
                            session_id, exc, code="audio_lag", close_code=1011
                        )
                        await sender; return
                    except RuntimeError as exc:
                        await runtime_value.enqueue_error(
                            session_id, exc, code="audio_limit", close_code=1008
                        )
                        await sender; return
                    continue
                try:
                    control = parse_client_command(json.loads(incoming.get("text") or "{}"))
                except (ValueError, json.JSONDecodeError) as exc:
                    await runtime_value.enqueue_error(
                        session_id, exc, code="invalid_command", close_code=1008
                    )
                    await sender; return
                if control.type == "segment":
                    await runtime_value.enqueue_segment(session_id)
                elif control.type == "abort":
                    await runtime_value.enqueue_abort(session_id)
                    await sender; return
                elif control.type == "finish":
                    await runtime_value.enqueue_finish(session_id)
                    await sender; return
                else:
                    raise ValueError("start may only be sent once")
        except (WebSocketDisconnect, TimeoutError):
            return
        except Exception as exc:
            if ctx is not None and protocol is not None and not protocol.terminal:
                with suppress(Exception):
                    await runtime_value.enqueue_error(session_id, exc)
                    if sender is not None:
                        await sender
        finally:
            try:
                if sender is not None:
                    sender.cancel()
                    with suppress(asyncio.CancelledError, WebSocketDisconnect):
                        await sender
            finally:
                await runtime_value.abort(session_id)

    return app


app = create_app()
