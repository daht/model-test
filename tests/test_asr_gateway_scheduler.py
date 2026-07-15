import asyncio
from dataclasses import replace

import pytest

from app.asr_gateway_backends import DispatchMode
from app.asr_gateway_scheduler import BatchKey, GatewayScheduler, InferenceJob, InferenceResult
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
        with pytest.raises(BufferError, match="ready queue"):
            scheduler.enqueue(job("overflow"))
        assert await scheduler.run_once("worker-1") == []
        clock.advance(.02)
        first = await scheduler.run_once("worker-1")
        assert len(first) == 1
        assert len(adapter.calls[0]) == 1
        await scheduler.run_once("worker-1", force=True)
        assert all(len({j.session_id for j in call}) == len(call) for call in adapter.calls)
        assert all(len({j.batch_key for j in call}) == 1 for call in adapter.calls)
    asyncio.run(scenario())


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
    assert cleaned == ["ok-1", "ok-2"]
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
    assert cleaned == ["cancelled-1", "bad-1"]
    assert published == []
    assert snapshot["queued_samples"] == 0
