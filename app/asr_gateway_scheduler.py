from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence

from app.asr_gateway_backends import DispatchMode, WorkerAdapter


class StaleResultError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchKey:
    worker_id: str
    model_revision: str
    language: str
    task: str
    timestamps: bool
    prompt_identity: str
    decoding_identity: str
    sample_format: str
    length_bucket: int


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    session_id: str
    generation: int
    job_sequence: int
    worker_id: str
    backend_session_id: str
    start_sample: int
    end_sample: int
    pcm: bytes
    deadline: float
    batch_key: BatchKey
    final: bool = False
    enqueued_at: float = 0.0
    queue_deadline: float = float("inf")

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    session_id: str
    generation: int
    job_sequence: int
    worker_id: str
    start_sample: int
    end_sample: int
    segment_id: int = 0
    text: str = ""
    confirmed_text: str = ""
    tail_text: str = ""
    final: bool = False
    error: str | None = None

    @classmethod
    def from_job(cls, job: InferenceJob, **values: Any) -> "InferenceResult":
        return cls(
            job_id=job.job_id, session_id=job.session_id, generation=job.generation,
            job_sequence=job.job_sequence, worker_id=job.worker_id,
            start_sample=job.start_sample, end_sample=job.end_sample, **values,
        )


AsyncHook = Callable[[Any], Awaitable[None]]


async def _noop(_: Any) -> None:
    return None


async def _noop_failure(_: str, __: str) -> None:
    return None


class GatewayScheduler:
    """Per-worker bounded EDF queues with deterministic single-iteration API."""

    def __init__(
        self,
        adapters: Mapping[str, WorkerAdapter],
        *,
        clock: Callable[[], float] = time.monotonic,
        max_wait_seconds: float,
        max_ready_jobs: int,
        max_queued_samples: int,
        cleanup: AsyncHook = _noop,
        publish: AsyncHook = _noop,
        worker_failed: Callable[[str, str], Awaitable[None]] = _noop_failure,
        stage_hook: Callable[[str, Sequence[InferenceJob], int], None] | None = None,
        reject: AsyncHook = _noop,
        inference_timeout_seconds: float | None = None,
    ) -> None:
        if max_wait_seconds < 0 or max_ready_jobs <= 0 or max_queued_samples <= 0:
            raise ValueError("scheduler bounds must be non-negative and finite")
        self.adapters = dict(adapters)
        self.clock = clock
        self.max_wait_seconds = max_wait_seconds
        self.max_ready_jobs = max_ready_jobs
        self.max_queued_samples = max_queued_samples
        self.cleanup = cleanup
        self.publish = publish
        self.worker_failed = worker_failed
        self.stage_hook = stage_hook
        self.reject = reject
        self.inference_timeout_seconds = inference_timeout_seconds
        self._queues: dict[str, list[InferenceJob]] = defaultdict(list)
        self._queued_samples = 0
        self._cancelled_generations: set[tuple[str, int]] = set()
        self._accepted: dict[tuple[str, int], asyncio.Event] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def enqueue(self, job: InferenceJob) -> None:
        if self._closed:
            raise RuntimeError("scheduler is closed")
        if job.worker_id not in self.adapters:
            raise KeyError(f"unknown worker_id {job.worker_id}")
        if job.sample_count <= 0 or len(job.pcm) != job.sample_count * 2:
            raise ValueError("job PCM range does not match pcm_s16le bytes")
        total_jobs = sum(len(queue) for queue in self._queues.values())
        if total_jobs >= self.max_ready_jobs:
            raise BufferError("ready queue job limit exceeded")
        if self._queued_samples + job.sample_count > self.max_queued_samples:
            raise BufferError("ready queue audio limit exceeded")
        if not job.enqueued_at:
            job = InferenceJob(**{**job.__dict__, "enqueued_at": self.clock()})
        self._queues[job.worker_id].append(job)
        self._queued_samples += job.sample_count
        self._wake.set()

    def cancel_session(self, session_id: str, *, generation: int) -> None:
        self._cancelled_generations.add((session_id, generation))

    async def wait_session_safe(self, session_id: str, *, generation: int) -> None:
        event = self._accepted.get((session_id, generation))
        if event is not None:
            await event.wait()

    async def run_once(self, worker_id: str, *, force: bool = False) -> list[InferenceResult]:
        queue = self._queues[worker_id]
        if not queue:
            return []
        expired = [item for item in queue if self.clock() > item.queue_deadline]
        for item in expired:
            queue.remove(item)
            self._queued_samples -= item.sample_count
            await self.reject(item)
            await self.publish(InferenceResult.from_job(item, error="queue_timeout"))
        if not queue:
            return []
        cancelled = [
            item for item in queue
            if (item.session_id, item.generation) in self._cancelled_generations
        ]
        for item in cancelled:
            queue.remove(item)
            self._queued_samples -= item.sample_count
            try:
                await self.reject(item)
            except Exception:
                pass
        if not queue:
            return []
        adapter = self.adapters[worker_id]
        caps = adapter.capabilities
        queue.sort(key=lambda item: (item.deadline, item.enqueued_at, item.job_id))
        first = queue[0]
        limit = 1 if caps.dispatch_mode is DispatchMode.SINGLE else caps.max_batch_items
        compatible = [
            item for item in queue
            if item.batch_key == first.batch_key
        ]
        due = self.clock() >= min(first.deadline, first.enqueued_at + self.max_wait_seconds)
        full = len({item.session_id for item in compatible}) >= limit
        if not (force or due or full):
            return []

        batch: list[InferenceJob] = []
        sessions: set[str] = set()
        cost = 0
        for item in compatible:
            if item.session_id in sessions:
                continue
            if len(batch) >= limit or cost + item.sample_count > caps.max_batch_samples:
                break
            batch.append(item)
            sessions.add(item.session_id)
            cost += item.sample_count
        if not batch:
            return []
        for item in batch:
            queue.remove(item)
            self._accepted.setdefault((item.session_id, item.generation), asyncio.Event())
        self._stage("scheduler_dispatched", batch, limit)
        self._stage("worker_accepted", batch, limit)
        try:
            self._stage("inference_started", batch, limit)
            submission = adapter.submit(batch)
            raw_results = (
                await asyncio.wait_for(submission, timeout=self.inference_timeout_seconds)
                if self.inference_timeout_seconds is not None
                else await submission
            )
            self._stage("inference_completed", batch, limit)
            if len(raw_results) != len(batch):
                raise RuntimeError("adapter result count does not match submitted batch")
            results = [self._coerce_result(job, result) for job, result in zip(batch, raw_results)]
        except Exception as exc:
            self._stage("inference_completed", batch, limit)
            await self.worker_failed(worker_id, "submit_failed")
            results = [InferenceResult.from_job(job, error=f"{type(exc).__name__}: batch failed") for job in batch]

        for job, result in zip(batch, results):
            # Ownership/queue cleanup is deliberately complete before publication.
            self._queued_samples -= job.sample_count
            try:
                await self.cleanup(job)
            except StaleResultError:
                self._mark_safe(job)
                continue
            except Exception:
                # A cleanup failure poisons this result; success is never visible.
                await self.worker_failed(worker_id, "cleanup_failed")
                await self.publish(InferenceResult.from_job(job, error="cleanup_failed"))
                self._mark_safe(job)
                continue
            if (job.session_id, job.generation) in self._cancelled_generations:
                self._mark_safe(job)
                continue
            if result.generation != job.generation or result.job_sequence != job.job_sequence:
                self._mark_safe(job)
                continue
            try:
                await self.publish(result)
            finally:
                self._mark_safe(job)
        return results

    def _stage(self, stage: str, jobs: Sequence[InferenceJob], capacity: int) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage, jobs, capacity)

    def _mark_safe(self, job: InferenceJob) -> None:
        key = (job.session_id, job.generation)
        event = self._accepted.pop(key, None)
        if event is not None:
            event.set()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="asr-gateway-scheduler")

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._closed:
            dispatched = False
            for worker_id in self.adapters:
                if await self.run_once(worker_id):
                    dispatched = True
            if dispatched:
                continue
            self._wake.clear()
            deadlines = [
                min(job.deadline, job.enqueued_at + self.max_wait_seconds)
                for queue in self._queues.values() for job in queue
            ]
            if not deadlines:
                await self._wake.wait()
                continue
            timeout = max(0.0, min(deadlines) - self.clock())
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except TimeoutError:
                pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "ready_depth": sum(len(q) for q in self._queues.values()),
            "queued_samples": self._queued_samples,
            "workers": {key: len(value) for key, value in self._queues.items()},
        }

    @staticmethod
    def _coerce_result(job: InferenceJob, result: Any) -> InferenceResult:
        if isinstance(result, InferenceResult):
            if (result.job_id, result.session_id, result.worker_id, result.start_sample, result.end_sample) != (
                job.job_id, job.session_id, job.worker_id, job.start_sample, job.end_sample
            ):
                return InferenceResult.from_job(job, error="invalid adapter result identity")
            return result
        raise TypeError("adapter returned malformed result")
