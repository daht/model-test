from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


class StreamingMode(str, Enum):
    STATEFUL = "stateful"
    CHUNKED = "chunked"
    ROLLING = "rolling"
    OFFLINE = "offline"


class DispatchMode(str, Enum):
    SINGLE = "single"
    FIXED_MICROBATCH = "fixed_microbatch"
    DYNAMIC_MICROBATCH = "dynamic_microbatch"
    CONTINUOUS = "continuous"
    STATEFUL_DYNAMIC = "stateful_dynamic"


class VadMode(str, Enum):
    NONE = "none"
    GATEWAY = "gateway"
    WORKER = "worker"
    BOTH = "both"


class ResultMode(str, Enum):
    CUMULATIVE_SNAPSHOT = "cumulative_snapshot"
    REPLACEABLE_SEGMENT = "replaceable_segment"
    CONFIRMED_PLUS_TAIL = "confirmed_plus_tail"


class WorkerLifecycle(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"


@dataclass(frozen=True)
class BackendCapabilities:
    protocol_version: int
    worker_id: str
    model_id: str
    model_revision: str
    gpu_id: str
    languages: tuple[str, ...]
    tasks: tuple[str, ...]
    streaming_mode: StreamingMode
    dispatch_mode: DispatchMode
    vad_mode: VadMode
    result_mode: ResultMode
    preferred_chunk_samples: int
    max_input_samples: int
    max_segment_samples: int
    max_batch_items: int
    max_batch_samples: int
    max_in_flight: int
    session_capacity: int
    retry_safe: bool
    warmed: bool = False
    sample_rate: int = 16_000
    sample_format: str = "pcm_s16le"
    backend_id: str = ""

    def __post_init__(self) -> None:
        for name in ("worker_id", "model_id", "model_revision", "gpu_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.backend_id and not self.backend_id.strip():
            raise ValueError("backend_id must not be blank")
        for name in (
            "protocol_version", "preferred_chunk_samples", "max_input_samples",
            "max_segment_samples",
            "max_batch_items", "max_batch_samples", "max_in_flight",
            "session_capacity", "sample_rate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.languages:
            raise ValueError("languages must not be empty")
        if not self.tasks:
            raise ValueError("tasks must not be empty")
        if self.dispatch_mode is DispatchMode.SINGLE and self.max_batch_items != 1:
            raise ValueError("max_batch_items must be one for single dispatch")
        if self.vad_mode is VadMode.BOTH:
            raise ValueError("vad_mode cannot enable gateway and worker ownership")
        if self.preferred_chunk_samples > self.max_input_samples:
            raise ValueError("preferred_chunk_samples exceeds max_input_samples")
        if self.max_input_samples > self.max_segment_samples:
            raise ValueError("max_input_samples exceeds max_segment_samples")

    @property
    def immutable_identity(self) -> tuple[str, str, str, str]:
        return self.worker_id, self.model_id, self.model_revision, self.gpu_id


@dataclass
class BackendSnapshot:
    capabilities: BackendCapabilities
    lifecycle: WorkerLifecycle = WorkerLifecycle.STARTING
    active_leases: int = 0
    queued_jobs: int = 0
    in_flight: int = 0
    load_error: str | None = None

    @property
    def accepting(self) -> bool:
        c = self.capabilities
        return (
            self.lifecycle is WorkerLifecycle.READY
            and c.warmed
            and self.active_leases < c.session_capacity
            and self.in_flight < c.max_in_flight
        )


class BackendLease:
    def __init__(self, registry: "BackendRegistry", worker_id: str) -> None:
        self._registry = registry
        self.worker_id = worker_id
        self._released = False
        self._lock = asyncio.Lock()

    async def release(self) -> None:
        async with self._lock:
            if self._released:
                return
            self._released = True
            await self._registry.release(self.worker_id)


class BackendRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, BackendSnapshot] = {}
        self._lock = asyncio.Lock()

    async def register(self, capabilities: BackendCapabilities) -> BackendSnapshot:
        async with self._lock:
            current = self._workers.get(capabilities.worker_id)
            if current is not None:
                if current.capabilities.immutable_identity != capabilities.immutable_identity:
                    raise ValueError("immutable model identity changed on re-registration")
                current.capabilities = capabilities
                return current
            for worker in self._workers.values():
                if worker.capabilities.gpu_id == capabilities.gpu_id:
                    raise ValueError(
                        f"gpu_id {capabilities.gpu_id} already has model owner "
                        f"{worker.capabilities.worker_id}"
                    )
            snapshot = BackendSnapshot(capabilities=capabilities)
            self._workers[capabilities.worker_id] = snapshot
            return snapshot

    async def mark_ready(self, worker_id: str, ready: bool, error: str | None = None) -> None:
        async with self._lock:
            worker = self._worker(worker_id)
            worker.lifecycle = WorkerLifecycle.READY if ready else WorkerLifecycle.FAILED
            worker.load_error = error

    async def acquire(
        self,
        *,
        preferred_worker_id: str | None = None,
        backend_id: str | None = None,
        language: str | None = None,
        task: str | None = None,
        streaming_mode: StreamingMode | None = None,
        result_modes: tuple[ResultMode, ...] | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
    ) -> BackendLease:
        async with self._lock:
            candidates = [w for w in self._workers.values() if w.accepting]
            if preferred_worker_id is not None:
                candidates = [w for w in candidates if w.capabilities.worker_id == preferred_worker_id]
            if backend_id is not None:
                candidates = [
                    w for w in candidates
                    if (w.capabilities.backend_id or w.capabilities.worker_id) == backend_id
                ]
            if language is not None and language != "auto":
                candidates = [
                    w for w in candidates
                    if language in w.capabilities.languages or "auto" in w.capabilities.languages
                ]
            if task is not None:
                candidates = [w for w in candidates if task in w.capabilities.tasks]
            if streaming_mode is not None:
                candidates = [w for w in candidates if w.capabilities.streaming_mode is streaming_mode]
            if result_modes is not None:
                candidates = [w for w in candidates if w.capabilities.result_mode in result_modes]
            if model_id is not None:
                candidates = [w for w in candidates if w.capabilities.model_id == model_id]
            if model_revision is not None:
                candidates = [w for w in candidates if w.capabilities.model_revision == model_revision]
            if not candidates:
                raise RuntimeError("unsupported backend route or no ready capacity")
            selected = min(candidates, key=lambda w: (w.active_leases, w.in_flight, w.capabilities.worker_id))
            selected.active_leases += 1
            return BackendLease(self, selected.capabilities.worker_id)

    async def release(self, worker_id: str) -> None:
        async with self._lock:
            worker = self._worker(worker_id)
            if worker.active_leases <= 0:
                raise RuntimeError("active lease accounting underflow")
            worker.active_leases -= 1

    async def begin_drain(self, worker_id: str) -> None:
        async with self._lock:
            self._worker(worker_id).lifecycle = WorkerLifecycle.DRAINING

    async def remove(self, worker_id: str) -> None:
        async with self._lock:
            worker = self._worker(worker_id)
            if worker.active_leases:
                raise RuntimeError("cannot remove worker with active leases")
            del self._workers[worker_id]

    async def get(self, worker_id: str) -> BackendSnapshot:
        async with self._lock:
            return self._worker(worker_id)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            workers = {
                worker_id: {
                    "worker_id": worker_id,
                    "model_id": item.capabilities.model_id,
                    "model_revision": item.capabilities.model_revision,
                    "gpu_id": item.capabilities.gpu_id,
                    "lifecycle": item.lifecycle.value,
                    "accepting": item.accepting,
                    "active_leases": item.active_leases,
                    "queued_jobs": item.queued_jobs,
                    "in_flight": item.in_flight,
                    "load_error": item.load_error,
                }
                for worker_id, item in self._workers.items()
            }
            return {"ready": any(w["accepting"] for w in workers.values()), "workers": workers}

    def _worker(self, worker_id: str) -> BackendSnapshot:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker_id {worker_id}") from exc


@dataclass(frozen=True)
class AdapterResult:
    backend_session_id: str
    text: str = ""
    confirmed_text: str = ""
    tail_text: str = ""
    decoded_samples: int = 0
    is_final: bool = False
    error: str | None = None
    metadata: Mapping[str, Any] | None = None


class WorkerAdapter(Protocol):
    capabilities: BackendCapabilities

    async def warmup(self) -> None: ...
    async def open_session(self, session_id: str, **options: Any) -> str: ...
    async def submit(self, jobs: Sequence[Any]) -> Sequence[Any]: ...
    async def finish_session(self, session_id: str) -> AdapterResult: ...
    async def abort_session(self, session_id: str) -> None: ...
    async def cancel(self, job_id: str) -> None: ...
    async def drain(self) -> None: ...
    async def close(self) -> None: ...
    async def snapshot(self) -> dict[str, Any]: ...


def sanitize_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: backend operation failed"
