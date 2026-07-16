import asyncio
from dataclasses import replace

from app.asr_gateway_backends import BackendRegistry
from app.asr_gateway_metrics import GatewayMetrics, JobTimeline, gateway_readiness
from tests.test_asr_gateway_backends import capabilities


def test_job_timeline_and_bounded_aggregates_are_sanitized():
    metrics = GatewayMetrics(max_completed=2)
    timeline = JobTimeline("j", "w", 16000)
    stages = ["audio_received", "chunk_ready", "scheduler_enqueued", "scheduler_dispatched", "worker_accepted", "inference_started", "inference_completed", "result_applied", "event_sent"]
    for index, stage in enumerate(stages): timeline.mark(stage, float(index))
    metrics.complete(timeline, batch_size=2, batch_capacity=4)
    metrics.set_gauges(active_sessions=3, ready_depth=2, queued_samples=8000, sample_rate=16000)
    snapshot = metrics.snapshot()
    assert snapshot["latency"]["chunk_wait_seconds"] == 1
    assert snapshot["latency"]["batch_wait_seconds"] == 1
    assert snapshot["latency"]["worker_wait_seconds"] == 1
    assert snapshot["latency"]["inference_seconds"] == 1
    assert snapshot["latency"]["egress_seconds"] == 1
    assert snapshot["decoded_seconds"] == 1
    assert snapshot["aggregate_rtf"] == 1
    assert snapshot["batch_fill_ratio"] == .5
    assert snapshot["active_sessions"] == 3
    forbidden = ("pcm", "authorization", "api_key", "transcript", "text")
    assert not any(word in str(snapshot).lower() for word in forbidden)


def test_completed_jobs_is_lifetime_total_not_bounded_window_size():
    metrics = GatewayMetrics(max_completed=2)

    for job_index in range(3):
        timeline = JobTimeline(f"j-{job_index}", "w", 16000)
        for stage_index, stage in enumerate(
            (
                "audio_received",
                "chunk_ready",
                "scheduler_enqueued",
                "scheduler_dispatched",
                "worker_accepted",
                "inference_started",
                "inference_completed",
                "result_applied",
                "event_sent",
            )
        ):
            timeline.mark(stage, float(stage_index))
        metrics.complete(timeline, batch_size=1, batch_capacity=1)

    snapshot = metrics.snapshot()

    assert snapshot["completed_jobs"] == 3
    assert snapshot["completed_window_jobs"] == 2


def test_readiness_requires_warmed_accepting_capacity():
    async def scenario():
        registry = BackendRegistry()
        assert await gateway_readiness(registry) is False
        await registry.register(replace(capabilities(), warmed=False))
        await registry.mark_ready("worker-1", True)
        assert await gateway_readiness(registry) is False
        await registry.register(capabilities())
        assert await gateway_readiness(registry) is True
        lease = await registry.acquire()
        second = await registry.acquire()
        assert await gateway_readiness(registry) is False
        await lease.release(); await second.release()
        await registry.begin_drain("worker-1")
        assert await gateway_readiness(registry) is False
    asyncio.run(scenario())
