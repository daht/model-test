import asyncio
from dataclasses import replace

import pytest

from app.asr_gateway_backends import DispatchMode
from app.asr_gateway_scheduler import BatchKey, GatewayScheduler, InferenceJob, InferenceResult, StaleResultError
from app.asr_observability import CapacityBufferError
from tests.test_asr_gateway_backends import capabilities


class FakeClock:
    def __init__(self): self.value = 0.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


class RecordingAdapter:
    def __init__(self, caps):
        self.capabilities = caps
        self.calls = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def submit(self, jobs):
        self.calls.append(list(jobs))
        self.entered.set()
        await self.release.wait()
        return [InferenceResult.from_job(job, text=job.session_id) for job in jobs]


def job(session_id, *, sequence=1, language="zh", samples=100, deadline=1.0):
    return InferenceJob(
        job_id=f"{session_id}-{sequence}", session_id=session_id, generation=1,
        job_sequence=sequence, worker_id="worker-1", backend_session_id=session_id,
        start_sample=(sequence - 1) * samples, end_sample=sequence * samples,
        pcm=b"\x00\x00" * samples, deadline=deadline,
        batch_key=BatchKey("worker-1", "rev-1", language, "transcribe", False, "", "default", "pcm_s16le", 0),
    )


def test_compatible_ready_sessions_form_one_dynamic_batch():
    async def scenario():
        clock = FakeClock()
        caps = replace(capabilities(), dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH, max_batch_items=4, max_batch_samples=1000)
        adapter = RecordingAdapter(caps)
        scheduler = GatewayScheduler({"worker-1": adapter}, clock=clock, max_wait_seconds=.02, max_ready_jobs=10, max_queued_samples=10_000)
        barrier = asyncio.Event()
        async def release(session):
            await barrier.wait(); scheduler.enqueue(job(session))
        producers = [asyncio.create_task(release(f"s{i}")) for i in range(3)]
        barrier.set(); await asyncio.gather(*producers)
        dispatch = asyncio.create_task(scheduler.run_once("worker-1", force=True))
        await adapter.entered.wait()
        assert [len(call) for call in adapter.calls] == [3]
        adapter.release.set(); await dispatch
    asyncio.run(scenario())


def test_serial_fallback_submits_lists_of_one():
    async def scenario():
        adapter = RecordingAdapter(capabilities())
        adapter.release.set()
        scheduler = GatewayScheduler({"worker-1": adapter}, clock=FakeClock(), max_wait_seconds=.02, max_ready_jobs=10, max_queued_samples=1000)
        scheduler.enqueue(job("a")); scheduler.enqueue(job("b"))
        await scheduler.run_once("worker-1", force=True)
        await scheduler.run_once("worker-1", force=True)
        return [len(call) for call in adapter.calls]
    assert asyncio.run(scenario()) == [1, 1]


def test_deadline_fairness_keys_and_bounds_without_sleep():
    async def scenario():
        clock = FakeClock()
        caps = replace(capabilities(), dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH, max_batch_items=3, max_batch_samples=500)
        adapter = RecordingAdapter(caps); adapter.release.set()
        scheduler = GatewayScheduler({"worker-1": adapter}, clock=clock, max_wait_seconds=.02, max_ready_jobs=3, max_queued_samples=350)
        scheduler.enqueue(job("same", sequence=1))
        scheduler.enqueue(job("same", sequence=2))
        scheduler.enqueue(job("other", language="ja"))
        with pytest.raises(CapacityBufferError, match="ready queue") as rejected:
            scheduler.enqueue(job("overflow"))
        assert rejected.value.reason == "scheduler_ready_job_limit"
        assert await scheduler.run_once("worker-1") == []
        clock.advance(.02)
        first = await scheduler.run_once("worker-1")
        assert len(first) == 1
        assert len(adapter.calls[0]) == 1
        await scheduler.run_once("worker-1", force=True)
        assert all(len({j.session_id for j in call}) == len(call) for call in adapter.calls)
        assert all(len({j.batch_key for j in call}) == 1 for call in adapter.calls)
    asyncio.run(scenario())


def test_due_final_batch_preempts_older_partial_batch():
    async def scenario():
        clock = FakeClock()
        clock.advance(2)
        caps = replace(
            capabilities(),
            dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH,
            max_batch_items=4,
            max_batch_samples=1000,
        )
        adapter = RecordingAdapter(caps)
        adapter.release.set()
        scheduler = GatewayScheduler(
            {"worker-1": adapter},
            clock=clock,
            max_wait_seconds=.2,
            max_ready_jobs=10,
            max_queued_samples=1000,
        )
        partial = job("partial", deadline=1.0)
        final = job("final", deadline=1.1)
        final = replace(
            final,
            final=True,
            batch_key=replace(final.batch_key, decoding_identity="final"),
        )
        scheduler.enqueue(partial)
        scheduler.enqueue(final)

        await scheduler.run_once("worker-1")

        return [[item.session_id for item in call] for call in adapter.calls]

    assert asyncio.run(scenario()) == [["final"]]


def test_scheduler_audio_overflow_has_exact_capacity_reason():
    scheduler = GatewayScheduler(
        {"worker-1": RecordingAdapter(capabilities())},
        clock=FakeClock(),
        max_wait_seconds=0,
        max_ready_jobs=10,
        max_queued_samples=150,
    )
    scheduler.enqueue(job("first", samples=100))

    with pytest.raises(CapacityBufferError, match="ready queue") as rejected:
        scheduler.enqueue(job("second", samples=100))

    assert rejected.value.reason == "scheduler_queued_audio_limit"
    assert rejected.value.safe_fields == {"limit": 150, "current": 100, "incoming": 100}


def test_completed_batch_releases_queued_audio_before_result_publication():
    async def scenario():
        caps = replace(
            capabilities(),
            dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH,
            max_batch_items=7,
            max_batch_samples=700,
        )
        adapter = RecordingAdapter(caps)
        adapter.release.set()
        rejected = []
        published = []
        scheduler = None

        async def publish(result):
            nonlocal scheduler
            published.append(result.job_id)
            if len(published) == 1:
                try:
                    scheduler.enqueue(job("followup", samples=200))
                except CapacityBufferError as exc:
                    rejected.append(exc.reason)

        scheduler = GatewayScheduler(
            {"worker-1": adapter},
            clock=FakeClock(),
            max_wait_seconds=0,
            max_ready_jobs=10,
            max_queued_samples=750,
            publish=publish,
        )
        for index in range(7):
            scheduler.enqueue(job(f"s{index}", samples=100))
        await scheduler.run_once("worker-1", force=True)
        return rejected, published, scheduler.snapshot()

    rejected, published, snapshot = asyncio.run(scenario())

    assert rejected == []
    assert len(published) == 7
    assert snapshot["queued_samples"] == 200


def test_cleanup_completes_before_success_publication_and_stale_is_discarded():
    async def scenario():
        adapter = RecordingAdapter(capabilities()); adapter.release.set()
        cleaned = []
        published = []
        async def cleanup(item): cleaned.append(item.job_id)
        async def publish(result):
            assert result.job_id in cleaned
            published.append(result.job_id)
        scheduler = GatewayScheduler({"worker-1": adapter}, clock=FakeClock(), max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000, cleanup=cleanup, publish=publish)
        scheduler.enqueue(job("ok"))
        await scheduler.run_once("worker-1", force=True)
        scheduler.cancel_session("ok", generation=1)
        stale = job("ok", sequence=2)
        scheduler.enqueue(stale)
        await scheduler.run_once("worker-1", force=True)
        return cleaned, published, scheduler.snapshot()
    cleaned, published, snapshot = asyncio.run(scenario())
    assert cleaned == ["ok-1"]
    assert published == ["ok-1"]
    assert snapshot["queued_samples"] == 0


def test_cancellation_before_acceptance_never_submits_and_cleanup_failure_never_publishes():
    async def scenario():
        adapter = RecordingAdapter(capabilities()); adapter.release.set()
        cleaned = []; published = []
        async def cleanup(item):
            cleaned.append(item.job_id)
            if item.session_id == "bad": raise RuntimeError("cleanup broke")
        async def publish(result): published.append(result.job_id)
        scheduler = GatewayScheduler({"worker-1": adapter}, clock=FakeClock(), max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000, cleanup=cleanup, publish=publish)
        scheduler.enqueue(job("cancelled")); scheduler.cancel_session("cancelled", generation=1)
        await scheduler.run_once("worker-1", force=True)
        scheduler.enqueue(job("bad")); await scheduler.run_once("worker-1", force=True)
        return adapter.calls, cleaned, published, scheduler.snapshot()
    calls, cleaned, published, snapshot = asyncio.run(scenario())
    assert calls == [[job("bad")]]
    assert cleaned == ["bad-1"]
    assert published == ["bad-1"]
    assert snapshot["queued_samples"] == 0


def test_worker_and_cleanup_failure_clear_readiness_but_stale_result_does_not():
    async def scenario():
        adapter = RecordingAdapter(capabilities()); adapter.release.set()
        failures = []; published = []
        async def failed(worker, reason): failures.append((worker, reason))
        async def cleanup(item):
            if item.session_id == "cleanup": raise RuntimeError("cleanup")
            if item.session_id == "stale": raise StaleResultError("stale")
        async def publish(result): published.append(result.job_id)
        scheduler = GatewayScheduler({"worker-1":adapter}, clock=FakeClock(), max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000, cleanup=cleanup, publish=publish, worker_failed=failed)
        scheduler.enqueue(job("cleanup")); await scheduler.run_once("worker-1", force=True)
        scheduler.enqueue(job("stale")); await scheduler.run_once("worker-1", force=True)
        async def broken(jobs): raise RuntimeError("worker lost")
        adapter.submit = broken
        scheduler.enqueue(job("worker")); await scheduler.run_once("worker-1", force=True)
        return failures, published
    failures, published = asyncio.run(scenario())
    assert [reason for _, reason in failures] == ["cleanup_failed", "submit_failed"]
    assert published == ["cleanup-1", "worker-1"]


def test_cancellation_after_worker_acceptance_waits_for_safe_completion():
    async def scenario():
        adapter = RecordingAdapter(capabilities())
        scheduler = GatewayScheduler({"worker-1":adapter}, clock=FakeClock(), max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000)
        scheduler.enqueue(job("accepted"))
        dispatch = asyncio.create_task(scheduler.run_once("worker-1", force=True))
        await adapter.entered.wait()
        scheduler.cancel_session("accepted", generation=1)
        barrier = asyncio.create_task(scheduler.wait_session_safe("accepted", generation=1))
        done, pending = await asyncio.wait({barrier}, timeout=0)
        assert not done and barrier in pending
        adapter.release.set(); await dispatch; await barrier
        return scheduler.snapshot()
    assert asyncio.run(scenario())["queued_samples"] == 0


def test_queue_deadline_expires_without_worker_acceptance():
    async def scenario():
        clock = FakeClock(); adapter = RecordingAdapter(capabilities()); adapter.release.set()
        rejected = []; published = []
        async def reject(item): rejected.append(item.job_id)
        async def publish(result): published.append(result)
        scheduler = GatewayScheduler({"worker-1":adapter}, clock=clock, max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000, reject=reject, publish=publish)
        expired = replace(job("late"), queue_deadline=.5)
        scheduler.enqueue(expired); clock.advance(.6)
        await scheduler.run_once("worker-1", force=True)
        return adapter.calls, rejected, published
    calls, rejected, published = asyncio.run(scenario())
    assert calls == [] and rejected == ["late-1"]
    assert published[0].error == "queue_timeout"


def test_inference_timeout_cancels_accepted_submit_and_marks_worker_failed():
    async def scenario():
        adapter = RecordingAdapter(capabilities())
        failures = []
        async def failed(worker, reason): failures.append(reason)
        scheduler = GatewayScheduler({"worker-1":adapter}, clock=FakeClock(), max_wait_seconds=0, max_ready_jobs=10, max_queued_samples=1000, inference_timeout_seconds=.001, worker_failed=failed)
        scheduler.enqueue(job("slow"))
        results = await scheduler.run_once("worker-1", force=True)
        await scheduler.wait_session_safe("slow", generation=1)
        return failures, results
    failures, results = asyncio.run(scenario())
    assert failures == ["submit_failed"]
    assert results[0].error == "TimeoutError: batch failed"


def test_submit_failure_halts_worker_and_settles_queued_jobs():
    class BrokenAdapter:
        capabilities = capabilities()

        def __init__(self):
            self.calls = []

        async def submit(self, jobs):
            self.calls.append([item.job_id for item in jobs])
            raise RuntimeError("private-worker-detail")

    async def scenario():
        adapter = BrokenAdapter()
        cleaned = []
        rejected = []
        published = []
        failures = []

        async def cleanup(item):
            cleaned.append(item.job_id)

        async def reject(item):
            rejected.append(item.job_id)

        async def publish(result):
            published.append(result)

        async def worker_failed(worker_id, reason):
            failures.append((worker_id, reason))

        scheduler = GatewayScheduler(
            {"worker-1": adapter},
            clock=FakeClock(),
            max_wait_seconds=0,
            max_ready_jobs=10,
            max_queued_samples=1000,
            cleanup=cleanup,
            reject=reject,
            publish=publish,
            worker_failed=worker_failed,
        )
        scheduler.enqueue(job("first"))
        scheduler.enqueue(job("queued"))
        await scheduler.run_once("worker-1", force=True)
        await scheduler.run_once("worker-1", force=True)
        return adapter, cleaned, rejected, published, failures, scheduler.snapshot()

    adapter, cleaned, rejected, published, failures, snapshot = asyncio.run(scenario())
    assert adapter.calls == [["first-1"]]
    assert failures == [("worker-1", "submit_failed")]
    assert sorted(cleaned + rejected) == ["first-1", "queued-1"]
    assert {item.job_id for item in published} == {"first-1", "queued-1"}
    assert all(item.error == "RuntimeError: batch failed" for item in published)
    assert snapshot["queued_samples"] == 0
    assert snapshot["ready_depth"] == 0
