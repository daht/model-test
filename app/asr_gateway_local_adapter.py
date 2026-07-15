from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from app.asr_gateway_backends import (
    AdapterResult, BackendCapabilities, DispatchMode, ResultMode,
    StreamingMode, VadMode, sanitize_error,
)
from app.asr_gateway_scheduler import InferenceJob, InferenceResult


class LocalCoordinatorAdapter:
    """Serial adapter; coordinator/model construction is deferred to warmup."""

    def __init__(
        self,
        coordinator_factory: Callable[[], Any],
        *,
        worker_id: str,
        model_id: str,
        model_revision: str,
        gpu_id: str,
        session_capacity: int = 2,
        max_input_samples: int = 480_000,
        vad_mode: VadMode = VadMode.GATEWAY,
    ) -> None:
        self._factory = coordinator_factory
        self._coordinator: Any | None = None
        self._sessions: dict[str, str] = {}
        self._ready = False
        self.capabilities = BackendCapabilities(
            protocol_version=1, worker_id=worker_id, model_id=model_id,
            model_revision=model_revision, gpu_id=gpu_id,
            languages=("auto", "zh", "ja", "en"), tasks=("transcribe",),
            streaming_mode=StreamingMode.STATEFUL, dispatch_mode=DispatchMode.SINGLE,
            vad_mode=vad_mode, result_mode=ResultMode.REPLACEABLE_SEGMENT,
            preferred_chunk_samples=24_000, max_input_samples=max_input_samples,
            max_batch_items=1, max_batch_samples=max_input_samples, max_in_flight=1,
            session_capacity=session_capacity, retry_safe=False, warmed=False,
        )

    async def warmup(self) -> None:
        if self._coordinator is None:
            self._coordinator = self._factory()
            await self._coordinator.start()
        snapshot = self._coordinator.snapshot()
        if not snapshot.ready or not snapshot.accepting:
            raise RuntimeError("coordinator warmup did not become ready")
        self._ready = True
        self.capabilities = replace(self.capabilities, warmed=True)

    async def open_session(self, session_id: str, *, language: str | None = None, **_: Any) -> str:
        coordinator = self._require_ready()
        if session_id in self._sessions:
            raise ValueError("duplicate gateway session")
        backend_id = await coordinator.create_stream(language)
        self._sessions[session_id] = backend_id
        return backend_id

    async def submit(self, jobs: Sequence[InferenceJob]) -> Sequence[InferenceResult]:
        if len(jobs) != 1:
            raise ValueError("serial adapter accepts exactly one item")
        coordinator = self._require_ready()
        job = jobs[0]
        backend_id = self._session(job.session_id)
        if backend_id != job.backend_session_id:
            raise KeyError("stale session backend identity")
        try:
            result = await coordinator.add_audio(backend_id, job.pcm, 16_000)
            return [InferenceResult.from_job(
                job, text=result.segment_text,
                confirmed_text="", tail_text=result.segment_text,
                final=bool(result.segment_finished),
            )]
        except Exception as exc:
            return [InferenceResult.from_job(job, error=sanitize_error(exc))]

    async def finish_segment(self, session_id: str) -> AdapterResult:
        coordinator = self._require_ready()
        backend_id = self._session(session_id)
        result = await coordinator.finish_segment(backend_id)
        return self._adapter_result(backend_id, result, final=False)

    async def finish_session(self, session_id: str) -> AdapterResult:
        coordinator = self._require_ready()
        backend_id = self._session(session_id)
        result = await coordinator.finish_stream(backend_id)
        self._sessions.pop(session_id, None)
        return self._adapter_result(backend_id, result, final=True)

    async def abort_session(self, session_id: str) -> None:
        backend_id = self._sessions.pop(session_id, None)
        if backend_id is not None and self._coordinator is not None:
            await self._coordinator.abort_stream(backend_id)

    async def cancel(self, job_id: str) -> None:
        # Coordinator enforces its own accepted/running cancellation barrier.
        return None

    async def drain(self) -> None:
        self._ready = False

    async def close(self) -> None:
        self._ready = False
        if self._coordinator is not None:
            for session_id in list(self._sessions):
                await self.abort_session(session_id)
            await self._coordinator.stop()
            self._coordinator = None

    async def snapshot(self) -> dict[str, Any]:
        if self._coordinator is None:
            return {"ready": False, "accepting": False, "active_sessions": 0, "capacity": self.capabilities.session_capacity}
        snapshot = self._coordinator.snapshot()
        return {
            "ready": bool(self._ready and snapshot.ready),
            "accepting": bool(self._ready and snapshot.accepting),
            "active_sessions": len(self._sessions),
            "capacity": self.capabilities.session_capacity,
            "queue_depth": snapshot.queue_depth,
            "queued_audio_seconds": snapshot.queued_audio_seconds,
            "load_error": snapshot.load_error,
        }

    def _require_ready(self) -> Any:
        if not self._ready or self._coordinator is None:
            raise RuntimeError("local coordinator adapter is not ready")
        return self._coordinator

    def _session(self, session_id: str) -> str:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError("stale session") from exc

    @staticmethod
    def _adapter_result(backend_id: str, result: Any, *, final: bool) -> AdapterResult:
        return AdapterResult(
            backend_session_id=backend_id, text=result.segment_text,
            tail_text=result.segment_text,
            decoded_samples=result.decoded_samples_delta or 0,
            is_final=final,
        )
