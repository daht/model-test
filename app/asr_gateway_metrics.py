from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from app.asr_gateway_backends import BackendRegistry
from app.asr_observability import BoundedValues


STAGES = (
    "audio_received", "chunk_ready", "scheduler_enqueued", "scheduler_dispatched",
    "worker_accepted", "inference_started", "inference_completed",
    "result_applied", "event_sent",
)


@dataclass
class JobTimeline:
    job_id: str
    worker_id: str
    decoded_samples: int
    sample_rate: int = 16_000
    stages: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, timestamp: float) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown metric stage {stage}")
        if self.stages and timestamp < max(self.stages.values()):
            raise ValueError("metric timestamps must be monotonic")
        self.stages[stage] = float(timestamp)

    def require_complete(self) -> None:
        missing = [stage for stage in STAGES if stage not in self.stages]
        if missing:
            raise ValueError(f"missing metric stages: {', '.join(missing)}")

    def durations(self) -> dict[str, float]:
        self.require_complete()
        s = self.stages
        return {
            "chunk_wait_seconds": s["chunk_ready"] - s["audio_received"],
            "batch_wait_seconds": s["scheduler_dispatched"] - s["scheduler_enqueued"],
            "worker_wait_seconds": s["inference_started"] - s["worker_accepted"],
            "inference_seconds": s["inference_completed"] - s["inference_started"],
            "egress_seconds": s["event_sent"] - s["result_applied"],
        }


class GatewayMetrics:
    def __init__(self, *, max_completed: int = 1024) -> None:
        if max_completed <= 0:
            raise ValueError("max_completed must be positive")
        self._completed: deque[tuple[JobTimeline, int, int]] = deque(maxlen=max_completed)
        self._completed_total = 0
        self._session_buffer_high_water_samples = 0
        self._scheduler_batch_sizes = BoundedValues(maxlen=max_completed)
        self._engine_group_sizes = BoundedValues(maxlen=max_completed)
        self._engine_groups_per_batch = BoundedValues(maxlen=max_completed)
        self._engine_partial_seconds = BoundedValues(maxlen=max_completed)
        self._engine_final_seconds = BoundedValues(maxlen=max_completed)
        self._engine_accumulated_audio_seconds = BoundedValues(maxlen=max_completed)
        self._engine_output_characters = BoundedValues(maxlen=max_completed)
        self._engine_maximum_character_run = BoundedValues(maxlen=max_completed)
        self._engine_calls = 0
        self._capacity_rejections: Counter[str] = Counter()
        self._gauges = {
            "active_sessions": 0,
            "ready_depth": 0,
            "queued_audio_seconds": 0.0,
            "session_buffered_audio_seconds": 0.0,
            "session_reserved_audio_seconds": 0.0,
            "max_session_held_audio_seconds": 0.0,
            "session_buffer_high_water_seconds": 0.0,
        }
        self.cancellations = 0
        self.conflicts = 0
        self.failures = 0

    def record_scheduler_batch(self, batch_size: int) -> None:
        self._scheduler_batch_sizes.add(batch_size)

    def record_capacity_rejection(self, reason: str) -> None:
        self._capacity_rejections[reason] += 1

    def record_engine_call(
        self,
        *,
        group_size: int,
        group_ordinal: int = 1,
        group_count: int = 1,
        elapsed_seconds: float,
        final: bool,
        accumulated_audio_seconds: float,
        output_characters: int,
        maximum_character_run: int,
    ) -> None:
        self._engine_calls += 1
        self._engine_group_sizes.add(group_size)
        if group_ordinal == 1:
            self._engine_groups_per_batch.add(group_count)
        target = self._engine_final_seconds if final else self._engine_partial_seconds
        target.add(elapsed_seconds)
        self._engine_accumulated_audio_seconds.add(accumulated_audio_seconds)
        self._engine_output_characters.add(output_characters)
        self._engine_maximum_character_run.add(maximum_character_run)

    def _observability_snapshot(self) -> dict[str, Any]:
        return {
            "scheduler_batch_size": self._scheduler_batch_sizes.summary(),
            "engine": {
                "calls": self._engine_calls,
                "group_size": self._engine_group_sizes.summary(),
                "groups_per_scheduler_batch": self._engine_groups_per_batch.summary(),
                "partial_inference_seconds": self._engine_partial_seconds.summary(),
                "final_inference_seconds": self._engine_final_seconds.summary(),
                "accumulated_audio_seconds": self._engine_accumulated_audio_seconds.summary(),
                "output_characters": self._engine_output_characters.summary(),
                "maximum_character_run": self._engine_maximum_character_run.summary(),
            },
            "buffer_rejections": dict(sorted(self._capacity_rejections.items())),
        }

    def complete(self, timeline: JobTimeline, *, batch_size: int, batch_capacity: int) -> None:
        timeline.require_complete()
        if batch_size <= 0 or batch_capacity < batch_size:
            raise ValueError("batch accounting is invalid")
        self._completed.append((timeline, batch_size, batch_capacity))
        self._completed_total += 1

    def set_gauges(
        self,
        *,
        active_sessions: int,
        ready_depth: int,
        queued_samples: int,
        session_buffered_samples: int,
        session_reserved_samples: int,
        max_session_held_samples: int,
        sample_rate: int,
    ) -> None:
        self._session_buffer_high_water_samples = max(
            self._session_buffer_high_water_samples,
            max(0, max_session_held_samples),
        )
        self._gauges = {
            "active_sessions": max(0, active_sessions),
            "ready_depth": max(0, ready_depth),
            "queued_audio_seconds": max(0, queued_samples) / sample_rate,
            "session_buffered_audio_seconds": (
                max(0, session_buffered_samples) / sample_rate
            ),
            "session_reserved_audio_seconds": (
                max(0, session_reserved_samples) / sample_rate
            ),
            "max_session_held_audio_seconds": (
                max(0, max_session_held_samples) / sample_rate
            ),
            "session_buffer_high_water_seconds": (
                self._session_buffer_high_water_samples / sample_rate
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        if not self._completed:
            return {
                **self._gauges,
                "completed_jobs": self._completed_total,
                "completed_window_jobs": 0,
                "cancellations": self.cancellations,
                "conflicts": self.conflicts,
                "failures": self.failures,
                **self._observability_snapshot(),
            }
        durations = [timeline.durations() for timeline, _, _ in self._completed]
        latency = {
            key: sum(item[key] for item in durations) / len(durations)
            for key in durations[0]
        }
        decoded_seconds = sum(t.decoded_samples / t.sample_rate for t, _, _ in self._completed)
        inference_seconds = sum(item["inference_seconds"] for item in durations)
        fill = sum(size / capacity for _, size, capacity in self._completed) / len(self._completed)
        return {
            **self._gauges,
            "completed_jobs": self._completed_total,
            "completed_window_jobs": len(self._completed),
            "decoded_seconds": decoded_seconds,
            "aggregate_rtf": inference_seconds / decoded_seconds if decoded_seconds else 0.0,
            "batch_fill_ratio": fill,
            "latency": latency,
            "cancellations": self.cancellations,
            "conflicts": self.conflicts,
            "failures": self.failures,
            **self._observability_snapshot(),
        }


async def gateway_readiness(registry: BackendRegistry) -> bool:
    return bool((await registry.snapshot())["ready"])
