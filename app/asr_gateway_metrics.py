from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.asr_gateway_backends import BackendRegistry


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
        self._gauges = {"active_sessions": 0, "ready_depth": 0, "queued_audio_seconds": 0.0}
        self.cancellations = 0
        self.conflicts = 0
        self.failures = 0

    def complete(self, timeline: JobTimeline, *, batch_size: int, batch_capacity: int) -> None:
        timeline.require_complete()
        if batch_size <= 0 or batch_capacity < batch_size:
            raise ValueError("batch accounting is invalid")
        self._completed.append((timeline, batch_size, batch_capacity))
        self._completed_total += 1

    def set_gauges(self, *, active_sessions: int, ready_depth: int, queued_samples: int, sample_rate: int) -> None:
        self._gauges = {
            "active_sessions": max(0, active_sessions),
            "ready_depth": max(0, ready_depth),
            "queued_audio_seconds": max(0, queued_samples) / sample_rate,
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
        }


async def gateway_readiness(registry: BackendRegistry) -> bool:
    return bool((await registry.snapshot())["ready"])
