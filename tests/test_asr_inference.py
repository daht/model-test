import asyncio
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
        await coordinator.stop()

        assert holder["transcriber"].sessions[0].abort_count == 1

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
