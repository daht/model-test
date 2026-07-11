import asyncio
import logging
import threading

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

    def reset_segment(self):
        self.owner.record("reset")

    def abort(self):
        self.owner.record("abort")
        self.abort_count += 1
        self.owner.abort_called.set()


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


def test_same_session_calls_complete_in_submission_order():
    async def scenario():
        coordinator, _records, holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream("en")
        results = await asyncio.gather(
            coordinator.add_audio(session_id, b"first", 16000),
            coordinator.add_audio(session_id, b"second", 16000),
        )
        await coordinator.stop()

        assert holder["transcriber"].sessions[0].chunks == [b"first", b"second"]
        assert results[-1].text == "firstsecond"

    asyncio.run(scenario())


def test_queue_full_raises_asr_queue_full():
    async def scenario():
        coordinator, _records, holder = make_coordinator(
            asr_inference_queue_size=1,
            asr_stream_inference_timeout_seconds=1.0,
        )
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        second = asyncio.create_task(coordinator.add_audio(session_id, b"second", 16000))
        await asyncio.sleep(0)
        with pytest.raises(ASRQueueFull):
            await coordinator.add_audio(session_id, b"third", 16000)
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
        session_id = await coordinator.create_stream(None)
        first = asyncio.create_task(coordinator.add_audio(session_id, b"block", 16000))
        assert await asyncio.to_thread(holder["transcriber"].call_started.wait, 1)
        with pytest.raises(ASRQueueTimeout):
            await coordinator.add_audio(session_id, b"expired", 16000)
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
            dict(coordinator._last_timings),
        )
        await coordinator.stop()

        assert holder["transcriber"].sessions[0].abort_count == 1
        assert registries == (set(), set(), {})

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


def test_finished_and_aborted_sessions_do_not_leave_timing_entries():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        finished_id = await coordinator.create_stream(None)
        await coordinator.finish_stream(finished_id)
        aborted_id = await coordinator.create_stream(None)
        await coordinator.abort_stream(aborted_id)
        timing_ids = set(coordinator._last_timings)
        await coordinator.stop()
        return timing_ids, finished_id, aborted_id

    timing_ids, finished_id, aborted_id = asyncio.run(scenario())

    assert finished_id not in timing_ids
    assert aborted_id not in timing_ids


def test_stop_and_restart_clear_all_session_registries():
    async def scenario():
        coordinator, _records, _holder = make_coordinator()
        await coordinator.start()
        session_id = await coordinator.create_stream(None)
        await coordinator.add_audio(session_id, b"timed", 16000)
        assert session_id in coordinator._last_timings
        await coordinator.stop()
        stopped_registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
            dict(coordinator._last_timings),
        )
        await coordinator.start()
        restarted_registries = (
            set(coordinator._poisoned_sessions),
            set(coordinator._active_sessions),
            dict(coordinator._last_timings),
        )
        await coordinator.stop()
        return stopped_registries, restarted_registries

    stopped, restarted = asyncio.run(scenario())

    assert stopped == (set(), set(), {})
    assert restarted == (set(), set(), {})


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
