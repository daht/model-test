import asyncio
import logging
import threading
from concurrent.futures import Future

import pytest

from app.asr import ASRStreamingSession, ASRTranscriber, StreamingTranscriptionResult, TranscriptionResult
from app.asr_inference import (
    ASRBatchConflict,
    ASRFileTranscriptionDisabled,
    ASRInferenceCoordinator,
    ASRInferenceTimeout,
    ASRNotReady,
    ASRQueueFull,
    ASRQueueTimeout,
    ASRSessionBusy,
    ASRSessionPoisoned,
)
from app.config import Settings


class FakeSession(ASRStreamingSession):
    def __init__(self, owner, language):
        self.owner = owner
        self.language = language
        self.chunks = []
        self.abort_count = 0

    def add_pcm_s16le(self, pcm_bytes, sample_rate):
        self.owner.record("add")
        if pcm_bytes == b"base-exit":
            raise SystemExit("fake owner-thread termination")
        if pcm_bytes == b"block":
            self.owner.call_started.set()
            self.owner.release_call.wait(2)
        if pcm_bytes == b"raise":
            raise RuntimeError("fake inference failure")
        self.chunks.append(pcm_bytes)
        return StreamingTranscriptionResult(text=b"".join(self.chunks).decode(), language=self.language)

    def finish(self):
        self.owner.record("finish")
        if getattr(self.owner, "finish_raises", False):
            raise RuntimeError("fake finish failure")
        return StreamingTranscriptionResult(text=b"".join(self.chunks).decode(), language=self.language)

    def finish_segment(self):
        self.owner.record("reset")
        return StreamingTranscriptionResult(
            text=b"".join(self.chunks).decode(),
            language=self.language,
        )

    def reset_segment(self):
        self.finish_segment()

    def abort(self):
        self.owner.record("abort")
        self.abort_count += 1
        self.owner.abort_called.set()
        if getattr(self.owner, "abort_raises", False):
            raise RuntimeError("fake abort failure")


class FakeTranscriber(ASRTranscriber):
    def __init__(self, records, *, block_file=False):
        self.records = records
        self.sessions = []
        self.block_file = block_file
        self.call_started = threading.Event()
        self.release_call = threading.Event()
        self.abort_called = threading.Event()
        self.file_started = threading.Event()
        self.release_file = threading.Event()
        self.record("constructor")

    def record(self, name):
        self.records.append((name, threading.get_ident()))

    def warmup(self):
        self.record("warmup")

    def create_streaming_session(self, language=None):
        self.record("create")
        session = FakeSession(self, language)
        self.sessions.append(session)
        return session

    def transcribe(self, audio_path, language=None):
        self.record("file")
        if self.block_file:
            self.file_started.set()
            self.release_file.wait(2)
        return TranscriptionResult(text="file result", language=language)


def settings(**overrides):
    values = {
        "asr_backend": "mock",
        "asr_eager_load": True,
        "asr_inference_queue_size": 4,
        "asr_max_active_streams": 2,
        "asr_max_queued_audio_seconds": 4.0,
        "asr_stream_queue_timeout_seconds": 0.2,
        "asr_stream_inference_timeout_seconds": 0.2,
        "asr_file_inference_timeout_seconds": 0.2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_coordinator(**setting_overrides):
    records = []
    holder = {}

    def factory():
        transcriber = FakeTranscriber(records)
        holder["transcriber"] = transcriber
        return transcriber

    coordinator = ASRInferenceCoordinator(settings(**setting_overrides), factory)
    return coordinator, records, holder


def test_model_construction_and_calls_share_owner_thread():
    async def scenario():
        coordinator, records, _holder = make_coordinator(asr_file_transcribe_enabled=True)
        await coordinator.start()
        session_id = await coordinator.create_stream("zh")
        await coordinator.add_audio(session_id, b"hello", 16000)
        await coordinator.reset_segment(session_id)
        await coordinator.finish_stream(session_id)
        aborted_id = await coordinator.create_stream("ja")
        await coordinator.abort_stream(aborted_id)
        await coordinator.transcribe_file("sample.wav", "en")
        await coordinator.stop()

        thread_ids = {thread_id for _name, thread_id in records}
        assert len(thread_ids) == 1
        assert thread_ids != {threading.get_ident()}

    asyncio.run(scenario())


def test_same_session_sequential_calls_complete_in_submission_order():
    async def scenario():
        coordinator, _records, holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream("en")
        first = await coordinator.add_audio(session_id, b"first", 16000)
        second = await coordinator.add_audio(session_id, b"second", 16000)
        await coordinator.stop()

        assert holder["transcriber"].sessions[0].chunks == [b"first", b"second"]
        assert first.text == "first"
        assert second.text == "firstsecond"

    asyncio.run(scenario())


def test_stream_result_carries_atomic_queue_and_inference_timings():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream("en")
        result = await coordinator.add_audio(session_id, b"first", 16000)
        await coordinator.stop()
        return coordinator, result

    coordinator, result = asyncio.run(scenario())

    assert result.queue_wait_seconds > 0
    assert result.inference_seconds > 0
    assert not hasattr(coordinator, "session_timing")


def test_queue_full_raises_asr_queue_full():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_inference_queue_size=1,
            asr_max_active_streams=3,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        first_id = await coordinator.create_stream(None)
        second_id = await coordinator.create_stream(None)
        third_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(first_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        second = asyncio.create_task(coordinator.add_audio(second_id, b"second", 16000))
        while coordinator.snapshot().queue_depth < 1:
            await asyncio.sleep(0)
        with pytest.raises(ASRQueueFull):
            await coordinator.add_audio(third_id, b"third", 16000)
        holder["transcriber"].release_call.set()
        await first
        await second
        await coordinator.stop()

    asyncio.run(scenario())


def test_expired_queued_job_never_calls_model():
    async def scenario():
        coordinator, records, holder = make_coordinator(
            asr_stream_queue_timeout_seconds=0.05,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        blocking_id = await coordinator.create_stream(None)
        expiring_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(blocking_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRQueueTimeout):
            await coordinator.add_audio(expiring_id, b"expired", 16000)
        holder["transcriber"].release_call.set()
        await first
        await coordinator.stop()

        assert [name for name, _thread_id in records].count("add") == 1

    asyncio.run(scenario())


def test_running_timeout_poisons_and_removes_session():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_queue_timeout_seconds=0.5,
            asr_stream_inference_timeout_seconds=0.05,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        task = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRInferenceTimeout):
            await task
        holder["transcriber"].release_call.set()
        assert await asyncio.to_thread(holder["transcriber"].abort_called.wait, 1)
        with pytest.raises(ASRSessionPoisoned):
            await coordinator.add_audio(session_id, b"later", 16000)
        barrier_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(barrier_id)
        registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
        )
        await coordinator.stop()

        assert holder["transcriber"].sessions[0].abort_count == 1
        assert registries == (set(), set())

    asyncio.run(scenario())


def test_worker_continues_after_one_job_raises():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        first_id = await coordinator.create_stream(None)
        second_id = await coordinator.create_stream(None)
        with pytest.raises(RuntimeError, match="fake inference failure"):
            await coordinator.add_audio(first_id, b"raise", 16000)
        result = await coordinator.add_audio(second_id, b"ok", 16000)
        await coordinator.stop()
        assert result.text == "ok"

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["add", "reset", "finish"])
def test_second_operation_for_busy_session_is_rejected_before_queueing(operation):
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        queue_depth_before = coordinator.snapshot().queue_depth
        with pytest.raises(ASRSessionBusy):
            if operation == "add":
                await coordinator.add_audio(session_id, b"second", 16000)
            elif operation == "reset":
                await coordinator.reset_segment(session_id)
            else:
                await coordinator.finish_stream(session_id)
        queue_depth_after = coordinator.snapshot().queue_depth
        holder["transcriber"].release_call.set()
        await first
        result = await coordinator.add_audio(session_id, b"later", 16000)
        await coordinator.abort_stream(session_id)
        await coordinator.stop()
        return queue_depth_before, queue_depth_after, result

    before, after, result = asyncio.run(scenario())

    assert before == after == 0
    assert result.text.endswith("later")


def test_queue_full_releases_session_operation_reservation():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_inference_queue_size=1,
            asr_max_active_streams=3,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        first_id = await coordinator.create_stream(None)
        queued_id = await coordinator.create_stream(None)
        rejected_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(first_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        queued = asyncio.create_task(coordinator.add_audio(queued_id, b"queued", 16000))
        while coordinator.snapshot().queue_depth < 1:
            await asyncio.sleep(0)
        with pytest.raises(ASRQueueFull):
            await coordinator.add_audio(rejected_id, b"rejected", 16000)
        holder["transcriber"].release_call.set()
        await first
        await queued
        retried = await coordinator.add_audio(rejected_id, b"retried", 16000)
        await coordinator.abort_stream(first_id)
        await coordinator.abort_stream(queued_id)
        await coordinator.abort_stream(rejected_id)
        await coordinator.stop()
        return retried

    result = asyncio.run(scenario())

    assert result.text == "retried"


def test_queue_timeout_reservation_releases_only_after_worker_discards_job():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_queue_timeout_seconds=0.05,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        blocking_id = await coordinator.create_stream(None)
        expiring_id = await coordinator.create_stream(None)
        blocking = asyncio.create_task(coordinator.add_audio(blocking_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRQueueTimeout):
            await coordinator.add_audio(expiring_id, b"expired", 16000)
        with pytest.raises(ASRSessionBusy):
            await coordinator.add_audio(expiring_id, b"too-soon", 16000)
        holder["transcriber"].release_call.set()
        await blocking
        await asyncio.to_thread(coordinator._jobs.join)
        retried = await coordinator.add_audio(expiring_id, b"retried", 16000)
        await coordinator.abort_stream(blocking_id)
        await coordinator.abort_stream(expiring_id)
        await coordinator.stop()
        return retried

    result = asyncio.run(scenario())

    assert result.text == "retried"


def test_abort_of_busy_timed_out_session_marks_poison_and_returns_immediately():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_inference_timeout_seconds=0.05,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        running = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRInferenceTimeout):
            await running
        await asyncio.wait_for(coordinator.abort_stream(session_id), timeout=0.1)
        poisoned_before_release = session_id in coordinator._poisoned_sessions
        holder["transcriber"].release_call.set()
        assert await asyncio.to_thread(holder["transcriber"].abort_called.wait, 1)
        barrier_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(barrier_id)
        cleaned = session_id not in coordinator._poisoned_sessions
        await coordinator.stop()
        return poisoned_before_release, cleaned

    poisoned, cleaned = asyncio.run(scenario())

    assert poisoned is True
    assert cleaned is True


def test_finish_failure_aborts_and_removes_session():
    async def scenario():
        coordinator, _records, holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        session = holder["transcriber"].sessions[0]
        holder["transcriber"].finish_raises = True
        with pytest.raises(RuntimeError, match="fake finish failure"):
            await coordinator.finish_stream(session_id)
        snapshot = coordinator.snapshot()
        await coordinator.stop()
        return session, snapshot

    session, snapshot = asyncio.run(scenario())

    assert session.abort_count == 1
    assert snapshot.active_streams == 0


def test_stop_rejects_new_jobs_and_joins_worker():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        await coordinator.stop()

        assert coordinator.worker_alive is False
        with pytest.raises(ASRNotReady):
            await coordinator.create_stream(None)

    asyncio.run(scenario())


def test_file_job_is_rejected_while_stream_is_active():
    async def scenario():
        coordinator, _records, _holder = make_coordinator(asr_file_transcribe_enabled=True)
        await coordinator.start()
        await coordinator.create_stream(None)
        with pytest.raises(ASRBatchConflict):
            await coordinator.transcribe_file("sample.wav", None)
        await coordinator.stop()

    asyncio.run(scenario())


def test_file_job_is_disabled_by_default():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        with pytest.raises(ASRFileTranscriptionDisabled):
            await coordinator.transcribe_file("sample.wav", None)
        await coordinator.stop()

    asyncio.run(scenario())


def test_stream_is_rejected_while_file_job_is_running():
    async def scenario():
        records = []
        holder = {}

        def factory():
            transcriber = FakeTranscriber(records, block_file=True)
            holder["transcriber"] = transcriber
            return transcriber

        coordinator = ASRInferenceCoordinator(
            settings(
                asr_file_transcribe_enabled=True,
                asr_file_inference_timeout_seconds=1.0,
            ),
            factory,
        )
        await coordinator.start()
        file_task = asyncio.create_task(coordinator.transcribe_file("sample.wav", None))
        assert await asyncio.to_thread(holder["transcriber"].file_started.wait, 1)
        with pytest.raises(ASRBatchConflict):
            await coordinator.create_stream(None)
        holder["transcriber"].release_file.set()
        await file_task
        await coordinator.stop()

    asyncio.run(scenario())


def test_operational_logs_include_timings_without_transcript_or_api_key(caplog):
    async def scenario():
        coordinator, _records, _holder = make_coordinator(api_key="distinctive-api-key")
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        await coordinator.add_audio(session_id, b"distinctive-transcript", 16000)
        await coordinator.finish_stream(session_id)
        await coordinator.stop()

    caplog.set_level(logging.INFO, logger="app.asr_inference")
    asyncio.run(scenario())
    messages = " ".join(record.getMessage() for record in caplog.records)

    assert "queue_wait_ms=" in messages
    assert "inference_ms=" in messages
    assert "distinctive-transcript" not in messages
    assert "distinctive-api-key" not in messages


def test_readiness_load_error_does_not_expose_exception_content():
    async def scenario():
        def factory():
            raise RuntimeError("/secret/model/path api_key=distinctive-api-key\nprivate")

        coordinator = ASRInferenceCoordinator(settings(), factory)
        await coordinator.start()
        snapshot = coordinator.snapshot()
        await coordinator.stop()
        return snapshot

    snapshot = asyncio.run(scenario())

    assert snapshot.ready is False
    assert snapshot.load_error == "RuntimeError: model warmup failed"


def test_coordinator_can_restart_cleanly_after_load_failure():
    async def scenario():
        records = []
        attempts = 0

        def factory():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("first load fails")
            return FakeTranscriber(records)

        coordinator = ASRInferenceCoordinator(settings(), factory)
        await coordinator.start()
        assert coordinator.snapshot().ready is False
        await coordinator.stop()
        await coordinator.start()
        snapshot = coordinator.snapshot()
        session_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(session_id)
        await coordinator.stop()
        return snapshot

    snapshot = asyncio.run(scenario())

    assert snapshot.ready is True
    assert snapshot.load_error is None


def test_lazy_warmup_failure_stops_without_leaking_worker_thread():
    async def scenario():
        records = []

        class FailingWarmupTranscriber(FakeTranscriber):
            def warmup(self):
                self.record("warmup")
                raise RuntimeError("lazy warmup failed")

        coordinator = ASRInferenceCoordinator(
            settings(asr_eager_load=False),
            lambda: FailingWarmupTranscriber(records),
        )
        await coordinator.start()
        with pytest.raises(RuntimeError, match="lazy warmup failed"):
            await coordinator.create_stream(None)
        snapshot = coordinator.snapshot()
        try:
            await asyncio.wait_for(coordinator.stop(), timeout=0.5)
        finally:
            if coordinator.worker_alive:
                shutdown = coordinator._new_job(
                    "shutdown",
                    (),
                    None,
                    priority=100,
                    queue_timeout=1.0,
                )
                coordinator._jobs.put_nowait(shutdown)
                await asyncio.to_thread(coordinator._thread.join, 1)
        return coordinator, snapshot

    coordinator, snapshot = asyncio.run(scenario())

    assert snapshot.accepting is False
    assert snapshot.load_error == "RuntimeError: model warmup failed"
    assert coordinator.worker_alive is False


def test_lazy_warmup_does_not_execute_job_that_expired_during_warmup():
    async def scenario():
        records = []
        warmup_started = threading.Event()
        release_warmup = threading.Event()

        class SlowWarmupTranscriber(FakeTranscriber):
            def warmup(self):
                self.record("warmup")
                warmup_started.set()
                release_warmup.wait(1)

        holder = {}

        def factory():
            transcriber = SlowWarmupTranscriber(records)
            holder["transcriber"] = transcriber
            return transcriber

        coordinator = ASRInferenceCoordinator(
            settings(
                asr_eager_load=False,
                asr_stream_queue_timeout_seconds=0.05,
                asr_stream_inference_timeout_seconds=1.0,
            ),
            factory,
        )
        await coordinator.start()
        create_task = asyncio.create_task(coordinator.create_stream(None))
        assert await asyncio.to_thread(warmup_started.wait, 1)
        with pytest.raises(ASRQueueTimeout):
            await create_task
        release_warmup.set()
        await asyncio.to_thread(coordinator._jobs.join)
        snapshot = coordinator.snapshot()
        session_count = len(holder["transcriber"].sessions)
        await coordinator.stop()
        return snapshot, session_count

    snapshot, session_count = asyncio.run(scenario())

    assert snapshot.active_streams == 0
    assert session_count == 0


@pytest.mark.parametrize("caller_exit", ["queue_timeout", "cancel"])
def test_create_stream_caller_exit_cannot_race_worker_claim_into_orphan(caller_exit):
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_queue_timeout_seconds=0.05,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()

        worker_passed_expiry_check = threading.Event()
        release_worker = threading.Event()
        original_expiry_check = coordinator._job_expired_before_start
        expiry_checks = 0

        def pause_after_validity_check(job):
            nonlocal expiry_checks
            expiry_checks += 1
            expired = original_expiry_check(job)
            if expiry_checks == 2 and not expired:
                worker_passed_expiry_check.set()
                release_worker.wait(1)
            return expired

        coordinator._job_expired_before_start = pause_after_validity_check

        original_new_job = coordinator._new_job
        caller_exit_started = threading.Event()

        class InterleavingCancellation:
            def __init__(self):
                self._event = threading.Event()

            def is_set(self):
                return self._event.is_set()

            def set(self):
                caller_exit_started.set()
                release_worker.set()
                holder["transcriber"].call_started.wait(0.3)
                self._event.set()

        def capture_job(*args, **kwargs):
            job = original_new_job(*args, **kwargs)
            if job.action == "create_stream":
                job.cancelled = InterleavingCancellation()
            return job

        coordinator._new_job = capture_job
        original_create = holder["transcriber"].create_streaming_session

        def recording_create(language=None):
            holder["transcriber"].call_started.set()
            return original_create(language)

        holder["transcriber"].create_streaming_session = recording_create

        create_task = asyncio.create_task(coordinator.create_stream(None))
        assert await asyncio.to_thread(worker_passed_expiry_check.wait, 1)
        if caller_exit == "cancel":
            create_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await create_task
        else:
            with pytest.raises(ASRQueueTimeout):
                await create_task
        assert caller_exit_started.is_set()
        await asyncio.to_thread(coordinator._jobs.join)
        snapshot = coordinator.snapshot()
        session_count = len(holder["transcriber"].sessions)
        poisoned = set(coordinator._poisoned_sessions)
        await coordinator.stop()
        return snapshot, session_count, poisoned

    snapshot, session_count, poisoned = asyncio.run(scenario())

    assert snapshot.active_streams == 0
    assert snapshot.queue_depth == 0
    assert snapshot.queued_audio_seconds == 0.0
    assert session_count == 0
    assert poisoned == set()


def test_poison_abort_failure_isolated_and_worker_remains_ready():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_stream_queue_timeout_seconds=0.5,
            asr_stream_inference_timeout_seconds=0.05,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        holder["transcriber"].abort_raises = True
        task = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRInferenceTimeout):
            await task
        holder["transcriber"].release_call.set()
        assert await asyncio.to_thread(holder["transcriber"].abort_called.wait, 1)
        await asyncio.sleep(0.02)
        snapshot = coordinator.snapshot()
        registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
        )
        worker_alive = coordinator.worker_alive
        if worker_alive:
            holder["transcriber"].abort_raises = False
            next_id = await coordinator.create_stream(None)
            await coordinator.abort_stream(next_id)
            await coordinator.stop()
        return snapshot, registries, worker_alive

    snapshot, registries, worker_alive = asyncio.run(scenario())

    assert worker_alive is True
    assert snapshot.ready is True
    assert snapshot.accepting is True
    assert registries == (set(), set())


def test_success_result_waits_for_worker_cleanup():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()

        submitted_job = {}
        original_new_job = coordinator._new_job

        def capture_job(*args, **kwargs):
            job = original_new_job(*args, **kwargs)
            if job.action == "create_stream":
                submitted_job["job"] = job
            return job

        coordinator._new_job = capture_job
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        original_cleanup = coordinator._cleanup_poisoned_session

        def blocking_cleanup(sessions, session_id):
            cleanup_started.set()
            assert release_cleanup.wait(1)
            original_cleanup(sessions, session_id)

        coordinator._cleanup_poisoned_session = blocking_cleanup
        create_task = asyncio.create_task(coordinator.create_stream(None))
        try:
            assert await asyncio.to_thread(cleanup_started.wait, 1)
            result_published_during_cleanup = submitted_job["job"].result.done()
        finally:
            release_cleanup.set()
        session_id = await create_task
        coordinator._cleanup_poisoned_session = original_cleanup
        await coordinator.abort_stream(session_id)
        await coordinator.stop()
        return result_published_during_cleanup

    result_published_during_cleanup = asyncio.run(scenario())

    assert result_published_during_cleanup is False


def test_unexpected_worker_exit_clears_cached_readiness():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        await asyncio.to_thread(coordinator._jobs.join)

        def raise_after_job(*_args):
            raise RuntimeError("unexpected cleanup failure")

        coordinator._cleanup_poisoned_session = raise_after_job
        with pytest.raises(ASRNotReady, match="owner worker stopped unexpectedly"):
            await coordinator.add_audio(session_id, b"ok", 16000)
        await asyncio.to_thread(coordinator._thread.join, 1)
        queue_joined = threading.Event()

        def join_queue():
            coordinator._jobs.join()
            queue_joined.set()

        threading.Thread(target=join_queue, daemon=True).start()
        accounting_completed = await asyncio.to_thread(queue_joined.wait, 0.2)
        return coordinator.snapshot(), coordinator.worker_alive, accounting_completed

    snapshot, worker_alive, accounting_completed = asyncio.run(scenario())

    assert worker_alive is False
    assert snapshot.ready is False
    assert snapshot.accepting is False
    assert accounting_completed is True


def test_base_exception_resolves_current_job_before_worker_terminates():
    async def scenario():
        coordinator, _records, _holder = make_coordinator(
            asr_stream_inference_timeout_seconds=0.5,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        started = asyncio.get_running_loop().time()
        with pytest.raises(ASRNotReady):
            await coordinator.add_audio(session_id, b"base-exit", 16000)
        elapsed = asyncio.get_running_loop().time() - started
        await asyncio.to_thread(coordinator._thread.join, 1)
        queue_joined = threading.Event()

        def join_queue():
            coordinator._jobs.join()
            queue_joined.set()

        threading.Thread(target=join_queue, daemon=True).start()
        accounting_completed = await asyncio.to_thread(queue_joined.wait, 0.2)
        return (
            elapsed,
            coordinator.snapshot(),
            coordinator.worker_alive,
            set(coordinator._busy_sessions),
            accounting_completed,
        )

    elapsed, snapshot, worker_alive, busy_sessions, accounting_completed = (
        asyncio.run(scenario())
    )

    assert elapsed < 0.2
    assert worker_alive is False
    assert snapshot.ready is False
    assert snapshot.accepting is False
    assert snapshot.queue_depth == 0
    assert snapshot.queued_audio_seconds == 0.0
    assert busy_sessions == set()
    assert accounting_completed is True


def test_finalizer_closes_admission_before_draining_queue():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        with coordinator._lock:
            coordinator._accepting = True
            coordinator._ready = True

        drain_started = threading.Event()
        release_drain = threading.Event()
        original_cancel_queued_jobs = coordinator._cancel_queued_jobs

        def blocking_cancel_queued_jobs(error):
            drain_started.set()
            release_drain.wait(1)
            original_cancel_queued_jobs(error)

        coordinator._cancel_queued_jobs = blocking_cancel_queued_jobs
        finalizer = threading.Thread(
            target=coordinator._finalize_worker,
            args=({},),
            daemon=True,
        )
        finalizer.start()
        assert await asyncio.to_thread(drain_started.wait, 1)

        admission = asyncio.create_task(coordinator.create_stream(None))
        await asyncio.sleep(0.02)
        rejected_before_drain = admission.done()
        queue_depth_during_drain = coordinator.snapshot().queue_depth

        release_drain.set()
        with pytest.raises(ASRNotReady):
            await admission
        await asyncio.to_thread(finalizer.join, 1)
        return rejected_before_drain, queue_depth_during_drain, coordinator.snapshot()

    rejected, queue_depth, snapshot = asyncio.run(scenario())

    assert rejected is True
    assert queue_depth == 0
    assert snapshot.accepting is False
    assert snapshot.ready is False


def test_finalizer_drains_jobs_before_aborting_sessions():
    class BlockingAbortSession:
        def __init__(self, abort_started, release_abort):
            self.abort_started = abort_started
            self.release_abort = release_abort

        def abort(self):
            self.abort_started.set()
            self.release_abort.wait(1)

    coordinator, _records, _holder = make_coordinator()
    with coordinator._lock:
        coordinator._accepting = True
        coordinator._ready = True
    drain_called = threading.Event()
    abort_started = threading.Event()
    release_abort = threading.Event()
    original_cancel_queued_jobs = coordinator._cancel_queued_jobs

    def recording_cancel_queued_jobs(error):
        drain_called.set()
        original_cancel_queued_jobs(error)

    coordinator._cancel_queued_jobs = recording_cancel_queued_jobs
    finalizer = threading.Thread(
        target=coordinator._finalize_worker,
        args=({"session": BlockingAbortSession(abort_started, release_abort)},),
        daemon=True,
    )
    finalizer.start()
    try:
        assert abort_started.wait(1)
        drained_before_abort_completed = drain_called.wait(0.05)
    finally:
        release_abort.set()
        finalizer.join(1)

    assert drained_before_abort_completed is True


def test_stop_drain_atomically_rejects_caller_cancelled_job_and_restarts():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_max_active_streams=2,
            asr_shutdown_grace_seconds=1.0,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        blocking_id = await coordinator.create_stream(None)
        queued_id = await coordinator.create_stream(None)

        drain_transition = threading.Event()
        caller_cancelled = threading.Event()
        captured = {}
        original_new_job = coordinator._new_job

        class InterleavingFuture(Future):
            def done(self):
                completed = super().done()
                if (
                    threading.current_thread().name == "asr-model-owner"
                    and not drain_transition.is_set()
                ):
                    drain_transition.set()
                    caller_cancelled.wait(1)
                return completed

            def set_running_or_notify_cancel(self):
                if (
                    threading.current_thread().name == "asr-model-owner"
                    and not drain_transition.is_set()
                ):
                    drain_transition.set()
                    caller_cancelled.wait(1)
                return super().set_running_or_notify_cancel()

        def capture_job(*args, **kwargs):
            job = original_new_job(*args, **kwargs)
            if job.action == "add_audio" and job.args[1] == b"queued":
                job.started = InterleavingFuture()
                captured["job"] = job
            return job

        coordinator._new_job = capture_job
        blocking = asyncio.create_task(coordinator.add_audio(blocking_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        queued = asyncio.create_task(coordinator.add_audio(queued_id, b"queued", 16000))
        while coordinator.snapshot().queue_depth < 1:
            await asyncio.sleep(0)
        stopping = asyncio.create_task(coordinator.stop())
        holder["transcriber"].release_call.set()
        await blocking
        assert await asyncio.to_thread(drain_transition.wait, 1)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        caller_cancelled.set()
        await stopping

        job = captured["job"]
        stopped_state = (
            job.started.done(),
            job.result.done(),
            coordinator._jobs.unfinished_tasks,
            coordinator.snapshot(),
            set(coordinator._busy_sessions),
        )
        await coordinator.start()
        restarted_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(restarted_id)
        await coordinator.stop()
        return stopped_state, coordinator.worker_alive

    stopped_state, worker_alive = asyncio.run(scenario())
    started_done, result_done, unfinished_tasks, snapshot, busy_sessions = stopped_state

    assert started_done is True
    assert result_done is True
    assert unfinished_tasks == 0
    assert snapshot.queue_depth == 0
    assert snapshot.queued_audio_seconds == 0.0
    assert busy_sessions == set()
    assert worker_alive is False


def test_stop_during_finalizer_abort_leaves_no_stale_control_job_and_restarts():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_shutdown_grace_seconds=1.0,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        session = holder["transcriber"].sessions[0]
        abort_started = threading.Event()
        release_abort = threading.Event()
        original_abort = session.abort

        def blocking_abort():
            abort_started.set()
            release_abort.wait(1)
            original_abort()

        session.abort = blocking_abort
        with pytest.raises(ASRNotReady):
            await coordinator.add_audio(session_id, b"base-exit", 16000)
        assert await asyncio.to_thread(abort_started.wait, 1)

        stop_task = asyncio.create_task(coordinator.stop())
        await asyncio.sleep(0.02)
        queued_during_finalizer = (
            coordinator._jobs.qsize(),
            coordinator._jobs.unfinished_tasks,
        )
        release_abort.set()
        await stop_task
        queued_after_stop = (
            coordinator._jobs.qsize(),
            coordinator._jobs.unfinished_tasks,
        )

        restarted = False
        if queued_after_stop == (0, 0):
            await coordinator.start()
            restarted_id = await coordinator.create_stream(None)
            await coordinator.abort_stream(restarted_id)
            await coordinator.stop()
            restarted = True
        return queued_during_finalizer, queued_after_stop, restarted

    during, after, restarted = asyncio.run(scenario())

    assert during == (0, 0)
    assert after == (0, 0)
    assert restarted is True


def test_concurrent_stop_enqueues_exactly_one_high_priority_control_job():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_shutdown_grace_seconds=1.0,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        running = asyncio.create_task(
            coordinator.add_audio(session_id, b"block", 16000)
        )
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)

        shutdown_priorities = []
        original_new_job = coordinator._new_job

        def recording_new_job(action, *args, **kwargs):
            if action == "shutdown":
                shutdown_priorities.append(kwargs["priority"])
            return original_new_job(action, *args, **kwargs)

        coordinator._new_job = recording_new_job
        first_stop = asyncio.create_task(coordinator.stop())
        second_stop = asyncio.create_task(coordinator.stop())
        await asyncio.sleep(0.02)
        holder["transcriber"].release_call.set()
        await running
        await asyncio.gather(first_stop, second_stop)
        return (
            shutdown_priorities,
            coordinator._jobs.qsize(),
            coordinator._jobs.unfinished_tasks,
            coordinator.worker_alive,
        )

    priorities, queue_size, unfinished, worker_alive = asyncio.run(scenario())

    assert priorities == [-100]
    assert queue_size == 0
    assert unfinished == 0
    assert worker_alive is False


def test_stop_during_lazy_warmup_never_creates_an_orphan_session():
    async def scenario():
        records = []
        warmup_started = threading.Event()
        release_warmup = threading.Event()

        class SlowWarmupTranscriber(FakeTranscriber):
            def warmup(self):
                self.record("warmup")
                warmup_started.set()
                release_warmup.wait(1)

        holder = {}

        def factory():
            transcriber = SlowWarmupTranscriber(records)
            holder["transcriber"] = transcriber
            return transcriber

        coordinator = ASRInferenceCoordinator(
            settings(
                asr_eager_load=False,
                asr_stream_queue_timeout_seconds=1.0,
                asr_shutdown_grace_seconds=1.0,
            ),
            factory,
        )
        await coordinator.start()
        create_task = asyncio.create_task(coordinator.create_stream(None))
        assert await asyncio.to_thread(warmup_started.wait, 1)
        stop_task = asyncio.create_task(coordinator.stop())
        while coordinator.snapshot().accepting:
            await asyncio.sleep(0)
        release_warmup.set()
        with pytest.raises(ASRNotReady):
            await create_task
        await stop_task
        return len(holder["transcriber"].sessions), coordinator.snapshot()

    session_count, snapshot = asyncio.run(scenario())

    assert session_count == 0
    assert snapshot.ready is False
    assert snapshot.accepting is False


def test_stop_is_bounded_when_running_call_blocks_and_business_queue_is_full():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_inference_queue_size=1,
            asr_max_active_streams=2,
            asr_stream_inference_timeout_seconds=2.0,
            asr_shutdown_grace_seconds=0.1,
        )
        await coordinator.start()
        first_id = await coordinator.create_stream(None)
        second_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(first_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        second = asyncio.create_task(coordinator.add_audio(second_id, b"queued", 16000))
        while coordinator.snapshot().queue_depth < 1:
            await asyncio.sleep(0)
        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(RuntimeError, match="did not stop"):
                await asyncio.wait_for(coordinator.stop(), timeout=0.5)
            elapsed = asyncio.get_running_loop().time() - started
        finally:
            holder["transcriber"].release_call.set()
        await first
        with pytest.raises(ASRNotReady):
            await second
        await asyncio.to_thread(coordinator._thread.join, 1)
        return elapsed, coordinator.worker_alive

    elapsed, worker_alive = asyncio.run(scenario())

    assert elapsed < 0.5
    assert worker_alive is False


def test_finished_and_aborted_sessions_need_no_timing_registry():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        finished_id = await coordinator.create_stream(None)
        await coordinator.finish_stream(finished_id)
        aborted_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(aborted_id)
        has_timing_registry = hasattr(coordinator, "_last_timings")
        await coordinator.stop()
        return has_timing_registry, finished_id, aborted_id

    has_timing_registry, finished_id, aborted_id = asyncio.run(scenario())

    assert has_timing_registry is False
    assert finished_id != aborted_id


def test_stop_and_restart_clear_all_session_registries():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        result = await coordinator.add_audio(session_id, b"timed", 16000)
        assert result.inference_seconds > 0
        await coordinator.stop()
        stopped_registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
        )
        await coordinator.start()
        restarted_registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
        )
        await coordinator.stop()
        return stopped_registries, restarted_registries

    stopped, restarted = asyncio.run(scenario())

    assert stopped == (set(), set())
    assert restarted == (set(), set())


def test_stream_chunk_transcription_runs_on_owner_thread_when_file_upload_is_disabled():
    async def scenario():
        coordinator, records, _holder = make_coordinator(asr_file_transcribe_enabled=False)
        await coordinator.start()
        session_id = await coordinator.create_chunked_stream("zh")
        assert coordinator.snapshot().active_streams == 1
        result = await coordinator.transcribe_stream_chunk(session_id, "chunk.wav", "zh", 1.0)
        await coordinator.abort_stream(session_id)
        assert coordinator.snapshot().active_streams == 0
        await coordinator.stop()
        return records, result

    records, result = asyncio.run(scenario())

    file_thread = next(thread_id for name, thread_id in records if name == "file")
    constructor_thread = next(thread_id for name, thread_id in records if name == "constructor")
    assert file_thread == constructor_thread
    assert result.text == "file result"
