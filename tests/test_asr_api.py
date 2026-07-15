import os
import logging
import threading
import asyncio
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

TEST_ONLY_LONG_API_KEY = "unit-test-only-not-a-production-secret-000000"

os.environ["API_KEY"] = "test-key"
os.environ["ASR_BACKEND"] = "mock"
os.environ["ASR_STREAM_CHUNK_SECONDS"] = "0.01"
os.environ["ASR_COMMIT_ON_PUNCTUATION"] = "false"
os.environ["ASR_VAD_SILENCE_SECONDS"] = "1.5"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

import app.asr_api as asr_api  # noqa: E402
from app.asr_api import app  # noqa: E402
from app.asr import TranscriptionResult  # noqa: E402
from app.asr_inference import (  # noqa: E402
    ASRFileTranscriptionDisabled,
    ASRQueueFull,
    CoordinatorSnapshot,
)
from app.config import Settings  # noqa: E402


client = TestClient(app)


class FakeCoordinator:
    def __init__(self, *, snapshot=None, result=None, error=None, release=None):
        self._snapshot = snapshot or CoordinatorSnapshot(True, True, 0, 0, 0.0, None)
        self.result = result or TranscriptionResult("coordinated", "en")
        self.error = error
        self.release = release
        self.transcribe_calls = []
        self.transcribe_started = threading.Event()

    def snapshot(self):
        return self._snapshot

    async def transcribe_file(self, path, language):
        self.transcribe_calls.append((path, language))
        self.transcribe_started.set()
        if self.error:
            raise self.error
        if self.release:
            await __import__("asyncio").to_thread(self.release.wait)
        return self.result


class RecordingWebSocket:
    def __init__(self):
        self.events = []
        self.close_codes = []

    async def send_json(self, payload):
        self.events.append(payload)

    async def close(self, code):
        self.close_codes.append(code)


class ProtocolCoordinator(FakeCoordinator):
    def __init__(
        self,
        updates=(),
        final_text="",
        add_error=None,
        create_error=None,
        reset_error=None,
        timing=(0.0, 0.0),
        timing_delay=0.0,
        operation_delays=None,
    ):
        super().__init__()
        self.updates = iter(updates)
        self.final_text = final_text
        self.add_error = add_error
        self.create_error = create_error
        self.reset_error = reset_error
        self.timing = timing
        self.timing_delay = timing_delay
        self.operation_delays = operation_delays or {}
        self.completed_operations = []
        self.abort_count = 0
        self.reset_count = 0
        self.latest_text = ""

    async def create_stream(self, _language):
        await self._delay("create")
        if self.create_error:
            raise self.create_error
        self.completed_operations.append("create")
        return "protocol-session"

    async def create_chunked_stream(self, _language):
        return await self.create_stream(_language)

    async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
        from app.asr import StreamingTranscriptionResult

        await self._delay("add")
        if self.add_error:
            raise self.add_error
        self.completed_operations.append("add")
        update = next(self.updates)
        if isinstance(update, StreamingTranscriptionResult):
            self.latest_text = update.text
            return self._with_timing(update)
        self.latest_text = update
        return self._with_timing(StreamingTranscriptionResult(update, "zh"))

    async def finish_stream(self, _session_id):
        from app.asr import StreamingTranscriptionResult

        await self._delay("finish")
        self.completed_operations.append("finish")
        return self._with_timing(
            StreamingTranscriptionResult(self.final_text or self.latest_text, "zh")
        )

    async def transcribe_stream_chunk(
        self,
        _session_id,
        _audio_path,
        _language,
        _audio_seconds,
    ):
        await self._delay("chunk")
        self.completed_operations.append("chunk")
        return TranscriptionResult(next(self.updates), "zh")

    async def reset_segment(self, _session_id):
        await self._delay("reset")
        if self.reset_error:
            raise self.reset_error
        self.completed_operations.append("reset")
        self.reset_count += 1
        from app.asr import StreamingTranscriptionResult

        return self._with_timing(
            StreamingTranscriptionResult(self.final_text or self.latest_text, "zh")
        )

    async def finish_segment(self, session_id):
        return await self.reset_segment(session_id)

    async def abort_stream(self, _session_id):
        self.abort_count += 1

    def session_timing(self, _session_id):
        if self.timing_delay:
            import time

            time.sleep(self.timing_delay)
        return self.timing

    def _with_timing(self, result):
        if self.timing_delay:
            import time

            time.sleep(self.timing_delay)
        return replace(
            result,
            queue_wait_seconds=self.timing[0],
            inference_seconds=self.timing[1],
        )

    async def _delay(self, operation):
        delay = self.operation_delays.get(operation, 0.0)
        if delay:
            await asyncio.sleep(delay)


def _protocol_settings(**overrides):
    values = {
        "api_key": "test-key",
        "asr_backend": "mock",
        "asr_stream_mode": "stateful",
        "asr_stable_commit_enabled": False,
        "asr_max_frame_bytes": 16000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _override_protocol(coordinator, **setting_overrides):
    current = _protocol_settings(**setting_overrides)
    app.dependency_overrides[asr_api.get_asr_coordinator] = lambda: coordinator
    app.dependency_overrides[get_settings] = lambda: current


def _assert_closed(websocket, code):
    with pytest.raises(WebSocketDisconnect) as exc_info:
        websocket.receive_json()
    assert exc_info.value.code == code


def _override_asr_dependencies(coordinator, **setting_overrides):
    current = Settings(
        _env_file=None,
        api_key="test-key",
        asr_backend="mock",
        **setting_overrides,
    )
    app.dependency_overrides[asr_api.get_asr_coordinator] = lambda: coordinator
    app.dependency_overrides[get_settings] = lambda: current


def _clear_asr_dependency_overrides():
    app.dependency_overrides.clear()


def _start_stream(websocket, *, expect_sequence=False):
    websocket.send_json(
        {
            "type": "start",
            "api_key": "test-key",
            "language": "zh",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        }
    )
    expected = {"type": "ready"}
    if expect_sequence:
        expected["sequence"] = 1
    event = websocket.receive_json()
    assert event == expected
    return event


def _send_transcribable_chunk(websocket):
    websocket.send_bytes(b"\x00\x00" * 160)


def _pcm_s16le_samples(value: int, sample_count: int) -> bytes:
    return int(value).to_bytes(2, byteorder="little", signed=True) * sample_count


def _monotonic_values(monkeypatch, values):
    iterator = iter(values)
    clock = type("Clock", (), {"monotonic": staticmethod(lambda: next(iterator))})
    monkeypatch.setattr(asr_api, "time", clock, raising=False)


def test_vad_segment_finish_flushes_buffered_audio_and_keeps_session_usable():
    from app.asr import StreamingTranscriptionResult

    class Coordinator(FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.add_results = iter(
                [
                    StreamingTranscriptionResult(
                        "", "zh", processed_samples=0, model_updated=False
                    ),
                    StreamingTranscriptionResult(
                        segment_id=1,
                        segment_text="继续",
                        language="zh",
                        decoded_samples_delta=1600,
                        model_updated=True,
                    ),
                ]
            )
            self.segment_finishes = 0

        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return next(self.add_results)

        async def finish_segment(self, _session_id):
            self.segment_finishes += 1
            return StreamingTranscriptionResult(
                    "尾音",
                    "zh",
                    processed_samples=14400,
                    model_updated=True,
                    segment_finished=True,
                    segment_id=0,
            )

        async def abort_stream(self, _session_id):
            return None

        def session_timing(self, _session_id):
            return (0.0, 0.0)

    async def scenario():
        websocket = RecordingWebSocket()
        coordinator = Coordinator()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(
                asr_vad_silence_seconds=0.8,
                asr_max_frame_bytes=28800,
                asr_ws_max_queue=2,
                asr_max_connection_lag_seconds=2.0,
            ),
            coordinator,
        )
        await controller.start("zh")
        await controller.add_audio(_pcm_s16le_samples(0, 14400))
        await controller.add_audio(_pcm_s16le_samples(1000, 1600))
        return websocket.events, coordinator.segment_finishes

    events, segment_finishes = asyncio.run(scenario())

    assert segment_finishes == 1
    assert [(event["type"], event.get("text")) for event in events] == [
        ("ready", None),
        ("partial", "尾音"),
        ("sentence_final", "尾音"),
        ("partial", ""),
        ("partial", "继续"),
    ]


def test_buffer_only_frame_does_not_advance_stable_punctuation():
    from app.asr import StreamingTranscriptionResult

    punctuated = "这是一个足够长的候选句子。"
    retracted = "这是一个足够长的候选句子"

    class Coordinator(FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.results = iter(
                [
                    StreamingTranscriptionResult(
                        punctuated, "zh", processed_samples=16000, model_updated=True
                    ),
                    StreamingTranscriptionResult(
                        punctuated, "zh", processed_samples=0, model_updated=False
                    ),
                    StreamingTranscriptionResult(
                        retracted, "zh", processed_samples=16000, model_updated=True
                    ),
                ]
            )

        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return next(self.results)

        async def abort_stream(self, _session_id):
            return None

        def session_timing(self, _session_id):
            return (0.0, 0.0)

    async def scenario():
        websocket = RecordingWebSocket()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(
                asr_stable_commit_enabled=True,
                asr_stable_commit_seconds=1.0,
                asr_stable_commit_min_chars=8,
                asr_stable_commit_min_updates=2,
                asr_max_frame_bytes=32000,
                asr_ws_max_queue=1,
            ),
            Coordinator(),
        )
        await controller.start("zh")
        frame = _pcm_s16le_samples(1000, 16000)
        await controller.add_audio(frame)
        events_after_decode = list(websocket.events)
        await controller.add_audio(frame)
        events_after_buffer = list(websocket.events)
        await controller.add_audio(frame)
        return controller, websocket.events, events_after_decode, events_after_buffer

    controller, events, after_decode, after_buffer = asyncio.run(scenario())

    assert after_buffer == after_decode
    assert not any(event["type"] == "sentence_final" for event in events)
    assert events[-1] == {"type": "partial", "text": retracted, "sequence": 3}
    assert controller.transcript.processed_samples == 32000
    assert controller.transcript.stable.candidate_updates == 0


def test_automatic_rollover_emits_ordered_segment_and_exact_final_text():
    from app.asr import StreamingTranscriptionResult

    class Coordinator(FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.results = iter(
                [
                    StreamingTranscriptionResult(
                        "甲乙",
                        "zh",
                        processed_samples=1600,
                        model_updated=True,
                        segment_finished=True,
                        continuation=StreamingTranscriptionResult(
                            segment_id=1,
                            segment_text="丙",
                            language="zh",
                            decoded_samples_delta=800,
                            model_updated=True,
                        ),
                    ),
                    StreamingTranscriptionResult(
                        segment_id=1,
                        segment_text="丙丁",
                        language="zh",
                        decoded_samples_delta=1600,
                        model_updated=True,
                    ),
                ]
            )

        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return next(self.results)

        async def finish_stream(self, _session_id):
            return StreamingTranscriptionResult(
                segment_id=1,
                segment_text="丙丁",
                language="zh",
                decoded_samples_delta=0,
                model_updated=False,
            )

        async def abort_stream(self, _session_id):
            return None

        def session_timing(self, _session_id):
            return (0.0, 0.0)

    async def scenario():
        websocket = RecordingWebSocket()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(),
            Coordinator(),
        )
        await controller.start("zh")
        frame = _pcm_s16le_samples(1000, 1600)
        await controller.add_audio(frame)
        await controller.add_audio(frame)
        await controller.finish()
        return websocket.events

    events = asyncio.run(scenario())

    assert [(event["type"], event.get("text")) for event in events] == [
        ("ready", None),
        ("partial", "甲乙"),
        ("sentence_final", "甲乙"),
        ("partial", ""),
        ("partial", "丙"),
        ("partial", "丙丁"),
        ("final", "丙丁"),
    ]
    reconstructed = "".join(
        event["text"] for event in events if event["type"] in {"sentence_final", "final"}
    )
    assert reconstructed == "甲乙丙丁"


def test_noop_finish_does_not_count_as_stable_punctuation_update():
    from app.asr import StreamingTranscriptionResult

    text = "这是一个足够长的候选句子。"

    class Coordinator(FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.results = iter(
                [
                    StreamingTranscriptionResult(text, "zh", 16000, True),
                    StreamingTranscriptionResult(text, "zh", 16000, True),
                ]
            )

        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return next(self.results)

        async def finish_stream(self, _session_id):
            return StreamingTranscriptionResult(text, "zh", 0, False)

        async def abort_stream(self, _session_id):
            return None

        def session_timing(self, _session_id):
            return (0.0, 0.0)

    async def scenario():
        websocket = RecordingWebSocket()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(
                asr_stable_commit_enabled=True,
                asr_stable_commit_seconds=0.1,
                asr_stable_commit_min_chars=8,
                asr_stable_commit_min_updates=3,
                asr_max_frame_bytes=32000,
                asr_ws_max_queue=1,
            ),
            Coordinator(),
        )
        await controller.start("zh")
        frame = _pcm_s16le_samples(1000, 16000)
        await controller.add_audio(frame)
        await controller.add_audio(frame)
        await controller.finish()
        return controller, websocket.events

    controller, events = asyncio.run(scenario())

    assert not any(event["type"] == "sentence_final" for event in events)
    assert events[-1] == {"type": "final", "text": text, "sequence": 3}
    assert controller.transcript.processed_samples == 32000


def test_finish_applies_real_flush_progress_before_final_event():
    from app.asr import StreamingTranscriptionResult

    class Coordinator(FakeCoordinator):
        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return StreamingTranscriptionResult("hello", "en", 16000, True)

        async def finish_stream(self, _session_id):
            return StreamingTranscriptionResult("hello world", "en", 8000, True)

        async def abort_stream(self, _session_id):
            return None

        def session_timing(self, _session_id):
            return (0.0, 0.0)

    async def scenario():
        websocket = RecordingWebSocket()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(
                asr_stable_commit_enabled=False,
                asr_max_frame_bytes=32000,
                asr_ws_max_queue=1,
            ),
            Coordinator(),
        )
        await controller.start("en")
        await controller.add_audio(_pcm_s16le_samples(1000, 16000))
        await controller.finish()
        return controller, websocket.events

    controller, events = asyncio.run(scenario())

    assert controller.transcript.processed_samples == 24000
    assert [(event["type"], event.get("text")) for event in events] == [
        ("ready", None),
        ("partial", "hello"),
        ("partial", "hello world"),
        ("final", "hello world"),
    ]


class FakeStreamingSession:
    def __init__(self, updates, final_text="hello world"):
        self.updates = iter(updates)
        self.finished = False
        self.final_text = final_text
        self.segment_reset = False

    def add_pcm_s16le(self, _pcm_bytes, _sample_rate):
        from app.asr import StreamingTranscriptionResult

        text = next(self.updates)
        return StreamingTranscriptionResult(text=text, language="zh")

    def finish(self):
        from app.asr import StreamingTranscriptionResult

        self.finished = True
        return StreamingTranscriptionResult(text=self.final_text, language="zh")

    def reset_segment(self):
        self.segment_reset = True

    def finish_segment(self):
        self.segment_reset = True
        from app.asr import StreamingTranscriptionResult

        return StreamingTranscriptionResult(self.final_text, "zh")


class FakeStatefulTranscriber:
    def __init__(self, session):
        self.session = session

    def create_streaming_session(self, language=None):
        self.language = language
        return self.session


class RaisingStatefulTranscriber:
    def __init__(self, exc):
        self.exc = exc

    def create_streaming_session(self, language=None):
        self.language = language
        raise self.exc


class RecordingStableCommitter:
    def __init__(self):
        self.enabled = True
        self.observations = []
        self.reset_count = 0

    def observe(self, text, now):
        self.observations.append((text, now))
        return None

    def reset(self):
        self.reset_count += 1


def test_asr_health_reports_model_name():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "Qwen3-ASR-1.7B"
    assert response.json()["backend"] == "mock"


def test_ready_returns_503_before_model_warmup():
    coordinator = FakeCoordinator(
        snapshot=CoordinatorSnapshot(False, True, 0, 0, 0.0, None)
    )
    _override_asr_dependencies(coordinator)
    try:
        response = client.get("/ready")
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ready_returns_snapshot_after_warmup():
    coordinator = FakeCoordinator(
        snapshot=CoordinatorSnapshot(True, True, 2, 3, 1.25, None)
    )
    _override_asr_dependencies(coordinator)
    try:
        response = client.get("/ready")
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "Qwen3-ASR-1.7B",
        "backend": "mock",
        "active_streams": 2,
        "queue_depth": 3,
        "queued_audio_seconds": 1.25,
        "detail": None,
    }


def test_file_transcribe_returns_503_when_disabled():
    coordinator = FakeCoordinator(error=ASRFileTranscriptionDisabled("disabled"))
    _override_asr_dependencies(coordinator, asr_file_transcribe_enabled=False)
    try:
        response = client.post(
            "/v1/transcribe",
            headers={"X-API-Key": "test-key"},
            files={"file": ("sample.wav", b"fake wav", "audio/wav")},
        )
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "file_transcription_disabled"


def test_file_transcribe_uses_async_coordinator_and_removes_upload():
    coordinator = FakeCoordinator(result=TranscriptionResult("hello", "en"))
    _override_asr_dependencies(coordinator, asr_file_transcribe_enabled=True)
    try:
        response = client.post(
            "/v1/transcribe",
            headers={"X-API-Key": "test-key"},
            data={"language": "en"},
            files={"file": ("sample.wav", b"fake wav", "audio/wav")},
        )
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    assert len(coordinator.transcribe_calls) == 1
    assert not Path(coordinator.transcribe_calls[0][0]).exists()


def test_health_remains_responsive_during_fake_slow_inference():
    release = threading.Event()
    coordinator = FakeCoordinator(release=release)
    _override_asr_dependencies(coordinator, asr_file_transcribe_enabled=True)
    response_holder = {}

    def transcribe_request():
        response_holder["response"] = client.post(
            "/v1/transcribe",
            headers={"X-API-Key": "test-key"},
            files={"file": ("sample.wav", b"fake wav", "audio/wav")},
        )

    thread = threading.Thread(target=transcribe_request)
    thread.start()
    try:
        assert coordinator.transcribe_started.wait(1)
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(1)
        _clear_asr_dependency_overrides()

    assert response_holder["response"].status_code == 200


def test_transcribe_rejects_missing_api_key():
    response = client.post(
        "/v1/transcribe",
        files={"file": ("sample.wav", b"fake wav", "audio/wav")},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_transcribe_accepts_audio_upload_with_language_hint():
    coordinator = FakeCoordinator(
        result=TranscriptionResult("[mock asr en] sample.wav", "en")
    )
    _override_asr_dependencies(coordinator, asr_file_transcribe_enabled=True)
    try:
        response = client.post(
            "/v1/transcribe",
            headers={"X-API-Key": "test-key"},
            data={"language": "en"},
            files={"file": ("sample.wav", b"fake wav", "audio/wav")},
        )
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 200
    assert response.json() == {
        "text": "[mock asr en] sample.wav",
        "language": "en",
        "model": "Qwen3-ASR-1.7B",
    }


def test_transcribe_rejects_unsupported_extension():
    _override_asr_dependencies(FakeCoordinator())
    try:
        response = client.post(
            "/v1/transcribe",
            headers={"X-API-Key": "test-key"},
            files={"file": ("sample.txt", b"not audio", "text/plain")},
        )
    finally:
        _clear_asr_dependency_overrides()

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported audio file type"


def test_stream_info_is_available_in_http_docs():
    response = client.get("/v1/transcribe/stream-info")

    assert response.status_code == 200
    body = response.json()
    assert body["websocket_url"] == "/v1/transcribe/stream"
    assert body["audio_format"]["format"] == "pcm_s16le"
    assert body["audio_format"]["vad_silence_seconds"] == 1.5
    assert body["start_message"]["type"] == "start"
    assert body["segment_message"] == {"type": "segment"}
    assert body["end_message"] == {"type": "end"}
    assert any(
        message["type"] == "sentence_final" and message["text"] == "..."
        for message in body["server_messages"]
    )


def test_stream_info_reports_punctuation_commit_setting():
    response = client.get("/v1/transcribe/stream-info")

    assert response.status_code == 200
    body = response.json()
    assert body["audio_format"]["commit_on_punctuation"] is False
    assert body["audio_format"]["vad_silence_seconds"] == 1.5


def test_stream_info_reports_streaming_mode_and_stateful_settings(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setenv("ASR_BACKEND", "qwen_vllm")
    monkeypatch.setenv("ASR_STREAM_CHUNK_SECONDS", "1.0")
    monkeypatch.setenv("ASR_STREAM_UNFIXED_CHUNK_NUM", "2")
    monkeypatch.setenv("ASR_STREAM_UNFIXED_TOKEN_NUM", "5")
    monkeypatch.setenv("ASR_VLLM_GPU_MEMORY_UTILIZATION", "0.8")
    monkeypatch.setenv("ASR_VLLM_MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("ASR_VLLM_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("ASR_STABLE_COMMIT_ENABLED", "true")
    monkeypatch.setenv("ASR_STABLE_COMMIT_SECONDS", "1.0")
    monkeypatch.setenv("ASR_STABLE_COMMIT_MIN_CHARS", "8")
    monkeypatch.setenv("ASR_STABLE_COMMIT_MIN_UPDATES", "2")
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    monkeypatch.setenv("API_KEY", TEST_ONLY_LONG_API_KEY)

    try:
        response = client.get("/v1/transcribe/stream-info")
    finally:
        asr_api.get_settings.cache_clear()

    assert response.status_code == 200
    body = response.json()
    assert body["audio_format"]["stream_mode"] == "stateful"
    assert body["audio_format"]["backend"] == "qwen_vllm"
    assert body["audio_format"]["vad_silence_seconds"] == 1.5
    assert body["audio_format"]["commit_on_punctuation"] is False
    assert body["audio_format"]["stateful"]["chunk_seconds"] == 1.0
    assert body["audio_format"]["stateful"]["unfixed_chunk_num"] == 2
    assert body["audio_format"]["stateful"]["unfixed_token_num"] == 5
    assert body["audio_format"]["stateful"]["vllm_gpu_memory_utilization"] == 0.8
    assert body["audio_format"]["stateful"]["vllm_max_model_len"] == 4096
    assert body["audio_format"]["stateful"]["vllm_max_new_tokens"] == 32
    assert body["audio_format"]["stateful"]["stable_commit_enabled"] is False
    assert body["audio_format"]["stateful"]["stable_commit_seconds"] == 1.0
    assert body["audio_format"]["stateful"]["stable_commit_min_chars"] == 8
    assert body["audio_format"]["stateful"]["stable_commit_min_updates"] == 2
    assert body["audio_format"]["commit_on_punctuation"] is False
    endpointing = body["audio_format"]["endpointing"]
    assert endpointing["backend"] == "silero_onnx_cpu"
    assert endpointing["immutable_commit_source"] == (
        "vad_endpoint_explicit_segment_or_forced_boundary"
    )
    assert endpointing["pre_roll_ms"] == 200
    assert endpointing["model_version"] == "6.2.1"
    assert endpointing["normal_utterance_seconds"] == 30.0
    assert endpointing["invariant_watchdog_seconds"] == 120.0


@pytest.mark.parametrize(
    ("raw_message", "code"),
    [
        ("not json", "invalid_start"),
        ('{"type":"start","api_key":"test-key","sample_rate":"bad"}', "invalid_start"),
        ('{"type":"start","api_key":"test-key","language":{}}', "invalid_language"),
    ],
)
def test_stream_v2_rejects_invalid_start_messages(raw_message, code):
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            websocket.send_text(raw_message)
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert event["code"] == code
            assert event["sequence"] == 1
            _assert_closed(websocket, 1003)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_rejects_unsupported_sample_rate():
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "api_key": "test-key",
                    "sample_rate": 8000,
                    "format": "pcm_s16le",
                }
            )
            event = websocket.receive_json()
            assert event["code"] == "unsupported_sample_rate"
            assert event["sequence"] == 1
            _assert_closed(websocket, 1003)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_rejects_bad_api_key_with_policy_close():
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "api_key": "bad-key",
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                }
            )
            event = websocket.receive_json()
            assert event["code"] == "invalid_api_key"
            assert event["sequence"] == 1
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()


@pytest.mark.parametrize(
    ("frame", "expected_code", "close_code"),
    [
        (b"", "invalid_audio_frame", 1003),
        (b"\x00", "invalid_audio_frame", 1003),
        (b"\x00\x00" * 17, "frame_too_large", 1009),
    ],
)
def test_stream_v2_validates_pcm_frames(frame, expected_code, close_code):
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator, asr_max_frame_bytes=32)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(frame)
            event = websocket.receive_json()
            assert event["code"] == expected_code
            assert event["sequence"] == 2
            _assert_closed(websocket, close_code)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_rejects_malformed_json_after_ready():
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_text("not json")
            event = websocket.receive_json()
            assert event["code"] == "invalid_message"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1003)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_maps_queue_overload_to_1013():
    coordinator = ProtocolCoordinator(add_error=ASRQueueFull("full"))
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00")
            event = websocket.receive_json()
            assert event["code"] == "server_busy"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1013)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_closes_when_realtime_lag_limit_is_exceeded():
    coordinator = ProtocolCoordinator(["late"], timing=(0.6, 0.5))
    _override_protocol(
        coordinator,
        asr_max_connection_lag_seconds=1.0,
        asr_max_undecoded_age_seconds=1.0,
        asr_ws_max_queue=2,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00")
            event = websocket.receive_json()
            assert event["code"] == "realtime_lag_exceeded"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1013)
        assert coordinator.abort_count == 1
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_accumulates_realtime_processing_debt():
    coordinator = ProtocolCoordinator(
        ["one", "two", "three", "four", "five"],
        timing=(0.0, 0.75),
    )
    _override_protocol(
        coordinator,
        asr_max_connection_lag_seconds=1.0,
        asr_max_undecoded_age_seconds=3.0,
        asr_ws_max_queue=2,
    )
    frame = _pcm_s16le_samples(1000, 8000)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            for _index in range(4):
                websocket.send_bytes(frame)
                assert websocket.receive_json()["type"] == "partial"
            websocket.send_bytes(frame)
            event = websocket.receive_json()
            assert event["code"] == "realtime_lag_exceeded"
            _assert_closed(websocket, 1013)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_enforces_cumulative_audio_limit():
    coordinator = ProtocolCoordinator(["first"])
    _override_protocol(coordinator, asr_max_audio_seconds=0.001)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 16)
            assert websocket.receive_json()["type"] == "partial"
            websocket.send_bytes(b"\x00\x00")
            event = websocket.receive_json()
            assert event["code"] == "audio_limit_exceeded"
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_v2_sequences_reconstruct_without_duplication():
    text = "这是一个足够长的稳定句子。"
    coordinator = ProtocolCoordinator([text, text, text], final_text=text)
    _override_protocol(
        coordinator,
        asr_stable_commit_enabled=True,
        asr_stable_commit_seconds=1.0,
        asr_stable_commit_min_updates=2,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(_pcm_s16le_samples(1000, 8000))
            first = websocket.receive_json()
            websocket.send_json({"type": "end"})
            final = websocket.receive_json()

            assert first == {"type": "partial", "text": text, "sequence": 2}
            assert final == {"type": "final", "text": text, "sequence": 3}
            _assert_closed(websocket, 1000)
    finally:
        _clear_asr_dependency_overrides()


def test_stateful_punctuation_snapshot_remains_replaceable_until_end():
    coordinator = ProtocolCoordinator(
        ["其中最。", "其中最重要的是保留模型修订。"],
        final_text="其中最重要的是保留模型修订。",
    )
    _override_protocol(
        coordinator,
        asr_commit_on_punctuation=True,
        asr_stable_commit_enabled=False,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(_pcm_s16le_samples(1000, 1600))
            assert websocket.receive_json() == {
                "type": "partial",
                "text": "其中最。",
                "sequence": 2,
            }
            websocket.send_bytes(_pcm_s16le_samples(1000, 1600))
            assert websocket.receive_json() == {
                "type": "partial",
                "text": "其中最重要的是保留模型修订。",
                "sequence": 3,
            }
            websocket.send_json({"type": "end"})
            assert websocket.receive_json() == {
                "type": "final",
                "text": "其中最重要的是保留模型修订。",
                "sequence": 4,
            }
            _assert_closed(websocket, 1000)
    finally:
        _clear_asr_dependency_overrides()


def test_vad_endpoint_flushes_once_and_requires_new_confirmed_speech_to_rearm():
    from collections import deque

    from app.asr import StreamingTranscriptionResult
    from app.asr_vad import StreamingVADEndpointDetector

    class Backend:
        def __init__(self):
            self.probabilities = deque(
                [0.9] * 3
                + [0.01] * 13
                + [0.9] * 3
                + [0.01] * 3
            )

        def speech_probability(self, _frame):
            return self.probabilities.popleft()

        def reset(self):
            return None

    vad = StreamingVADEndpointDetector(
        backend=Backend(),
        sample_rate=16000,
        frame_samples=512,
        onset_threshold=0.65,
        offset_threshold=0.35,
        min_speech_ms=96,
        min_silence_ms=96,
        hangover_ms=32,
        pre_roll_ms=200,
    )

    class Coordinator(FakeCoordinator):
        def __init__(self):
            super().__init__()
            self.add_results = iter(
                [
                    StreamingTranscriptionResult(
                        segment_id=0,
                        segment_text="其中最。",
                        language="zh",
                        decoded_samples_delta=1536,
                    ),
                    StreamingTranscriptionResult(
                        segment_id=0,
                        segment_text="其中最重要",
                        language="zh",
                        decoded_samples_delta=512,
                    ),
                    StreamingTranscriptionResult(
                        segment_id=1,
                        segment_text="第二句",
                        language="zh",
                        decoded_samples_delta=1536,
                    ),
                    StreamingTranscriptionResult(
                        segment_id=1,
                        segment_text="第二句",
                        language="zh",
                        decoded_samples_delta=512,
                    ),
                ]
            )
            self.segment_results = iter(
                [
                    StreamingTranscriptionResult(
                        segment_id=0,
                        segment_text="其中最重要的。",
                        language="zh",
                        decoded_samples_delta=0,
                        segment_finished=True,
                    ),
                    StreamingTranscriptionResult(
                        segment_id=1,
                        segment_text="第二句。",
                        language="zh",
                        decoded_samples_delta=0,
                        segment_finished=True,
                    ),
                ]
            )
            self.add_calls = 0
            self.segment_calls = 0

        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            self.add_calls += 1
            return next(self.add_results)

        async def finish_segment(self, _session_id):
            self.segment_calls += 1
            return next(self.segment_results)

        async def finish_stream(self, _session_id):
            return StreamingTranscriptionResult(
                segment_id=2,
                segment_text="",
                language="zh",
                decoded_samples_delta=0,
                model_updated=False,
            )

        async def abort_stream(self, _session_id):
            return None

    async def scenario():
        websocket = RecordingWebSocket()
        coordinator = Coordinator()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(),
            coordinator,
            vad_detector=vad,
        )
        await controller.start("zh")
        for probability_index in range(22):
            is_speech = probability_index < 3 or 16 <= probability_index < 19
            await controller.add_audio(
                _pcm_s16le_samples(1000 if is_speech else 0, 512)
            )
        await controller.finish()
        return websocket.events, coordinator.add_calls, coordinator.segment_calls

    events, add_calls, segment_calls = asyncio.run(scenario())

    assert [
        event["text"] for event in events if event["type"] == "sentence_final"
    ] == ["其中最重要的。", "第二句。"]
    assert [event for event in events if event["type"] == "final"] == [
        {"type": "final", "text": "", "sequence": events[-1]["sequence"]}
    ]
    assert add_calls == 4
    assert segment_calls == 2
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )


@pytest.mark.parametrize(
    ("frame_samples", "frame_count", "inference_seconds"),
    [(3200, 10, 0.10), (8000, 4, 0.25)],
)
def test_chunk_buffering_lag_uses_decoded_progress_not_transport_frames(
    frame_samples, frame_count, inference_seconds
):
    from app.asr import StreamingTranscriptionResult

    controller = asr_api.StreamingSessionController(
        RecordingWebSocket(),
        _protocol_settings(),
        FakeCoordinator(),
    )

    decisions = []
    for index in range(frame_count):
        decisions.append(
            controller._observe_stream_progress(
                StreamingTranscriptionResult(
                    segment_id=0,
                    segment_text="",
                    decoded_samples_delta=(
                        32000 if index == frame_count - 1 else 0
                    ),
                    model_updated=index == frame_count - 1,
                    inference_seconds=inference_seconds,
                ),
                submitted_samples=frame_samples,
            )
        )

    assert decisions == [False] * frame_count
    assert controller.processing_debt_seconds == pytest.approx(0.0)
    assert controller.oldest_undecoded_age_seconds == 0.0


def test_sustained_real_lag_requires_debt_and_oldest_audio_age_and_fires_once():
    from app.asr import StreamingTranscriptionResult

    controller = asr_api.StreamingSessionController(
        RecordingWebSocket(),
        _protocol_settings(
            asr_max_connection_lag_seconds=1.0,
            asr_max_undecoded_age_seconds=2.0,
            asr_ws_max_queue=1,
        ),
        FakeCoordinator(),
    )
    result = StreamingTranscriptionResult(
        segment_id=0,
        segment_text="",
        decoded_samples_delta=0,
        model_updated=False,
        queue_wait_seconds=0.25,
        inference_seconds=0.5,
    )

    decisions = [
        controller._observe_stream_progress(result, submitted_samples=3200)
        for _ in range(6)
    ]

    assert decisions == [False, False, True, False, False, False]
    assert controller.processing_debt_seconds == pytest.approx(4.5)
    assert controller.oldest_undecoded_age_seconds >= 4.5


def test_repeated_slow_fully_decoded_calls_trigger_sustained_lag_once():
    from app.asr import StreamingTranscriptionResult

    controller = asr_api.StreamingSessionController(
        RecordingWebSocket(),
        _protocol_settings(),
        FakeCoordinator(),
    )
    result = StreamingTranscriptionResult(
        segment_id=0,
        segment_text="snapshot",
        decoded_samples_delta=3200,
        model_updated=True,
        inference_seconds=1.0,
    )

    decisions = [
        controller._observe_stream_progress(result, submitted_samples=3200)
        for _ in range(6)
    ]

    assert decisions == [False, False, True, False, False, False]
    assert controller.processing_debt_seconds == pytest.approx(4.8)
    assert controller.oldest_undecoded_age_seconds == 0.0
    assert controller.sustained_decoded_lag_seconds == pytest.approx(4.8)


def test_slow_endpoint_flush_counts_toward_sustained_decoded_lag():
    from app.asr import StreamingTranscriptionResult

    controller = asr_api.StreamingSessionController(
        RecordingWebSocket(),
        _protocol_settings(),
        FakeCoordinator(),
    )
    buffered = StreamingTranscriptionResult(
        segment_id=0,
        segment_text="",
        decoded_samples_delta=0,
        model_updated=False,
        inference_seconds=0.1,
    )
    endpoint_flush = StreamingTranscriptionResult(
        segment_id=0,
        segment_text="final snapshot",
        decoded_samples_delta=3200,
        model_updated=True,
        segment_finished=True,
        inference_seconds=3.0,
    )

    assert (
        controller._observe_stream_progress(buffered, submitted_samples=3200)
        is False
    )
    assert (
        controller._observe_stream_progress(endpoint_flush, submitted_samples=0)
        is True
    )


def test_slow_endpoint_flush_emits_one_realtime_lag_error():
    from app.asr import StreamingTranscriptionResult

    class Coordinator(FakeCoordinator):
        async def create_stream(self, _language):
            return "session"

        async def add_audio(self, _session_id, _pcm_bytes, _sample_rate):
            return StreamingTranscriptionResult(
                segment_id=0,
                segment_text="",
                decoded_samples_delta=0,
                model_updated=False,
                inference_seconds=0.1,
            )

        async def finish_segment(self, _session_id):
            return StreamingTranscriptionResult(
                segment_id=0,
                segment_text="final snapshot",
                decoded_samples_delta=3200,
                model_updated=True,
                segment_finished=True,
                inference_seconds=3.0,
            )

        async def abort_stream(self, _session_id):
            return None

    async def scenario():
        websocket = RecordingWebSocket()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(),
            Coordinator(),
        )
        await controller.start("zh")
        await controller.add_audio(_pcm_s16le_samples(1000, 3200))
        with pytest.raises(asr_api._StreamClosed):
            await controller.reset_segment()
        return websocket

    websocket = asyncio.run(scenario())

    errors = [event for event in websocket.events if event["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "realtime_lag_exceeded"
    assert websocket.close_codes == [1013]


def test_stream_v2_new_segment_snapshot_does_not_conflict_with_frozen_text():
    coordinator = ProtocolCoordinator(["confirmed", "confirmed", "unsafe"])
    _override_protocol(
        coordinator,
        asr_vad_silence_seconds=0.001,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json()["type"] == "partial"
            websocket.send_bytes(_pcm_s16le_samples(0, 160))
            assert websocket.receive_json()["type"] == "sentence_final"
            assert websocket.receive_json() == {"type": "partial", "text": "", "sequence": 4}
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            event = websocket.receive_json()
            assert event == {"type": "partial", "text": "unsafe", "sequence": 5}
            websocket.send_json({"type": "end"})
            assert websocket.receive_json() == {
                "type": "final",
                "text": "unsafe",
                "sequence": 6,
            }
            _assert_closed(websocket, 1000)
    finally:
        _clear_asr_dependency_overrides()


def test_stream_revision_logs_neither_transcript_nor_api_key(caplog):
    coordinator = ProtocolCoordinator(["distinctive-transcript", "distinctive-transcript", "unsafe"])
    _override_protocol(coordinator, asr_vad_silence_seconds=0.001)
    caplog.set_level(logging.WARNING, logger="app.asr_api")
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            websocket.receive_json()
            websocket.send_bytes(_pcm_s16le_samples(0, 160))
            websocket.receive_json()
            websocket.receive_json()
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            websocket.receive_json()
    finally:
        _clear_asr_dependency_overrides()

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "distinctive-transcript" not in messages
    assert "test-key" not in messages


def test_stream_v2_enforces_session_timeout():
    coordinator = ProtocolCoordinator()
    _override_protocol(coordinator, asr_max_session_seconds=0.01, asr_idle_timeout_seconds=1.0)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            event = websocket.receive_json()
            assert event["code"] == "session_timeout"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()


@pytest.mark.parametrize("operation", ["add", "reset", "finish"])
def test_stateful_session_deadline_cancels_model_operation_before_late_event(operation):
    coordinator = ProtocolCoordinator(
        ["late partial"],
        final_text="late final",
        operation_delays={operation: 0.08},
    )
    _override_protocol(
        coordinator,
        asr_max_session_seconds=0.03,
        asr_idle_timeout_seconds=1.0,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            if operation == "add":
                websocket.send_bytes(b"\x00\x00" * 16)
            elif operation == "reset":
                websocket.send_json({"type": "segment"})
            else:
                websocket.send_json({"type": "end"})
            event = websocket.receive_json()
            assert event == {
                "type": "error",
                "code": "session_timeout",
                "message": "Maximum session duration exceeded",
                "sequence": 2,
            }
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()

    assert operation not in coordinator.completed_operations
    assert coordinator.abort_count >= 1


def test_chunked_session_deadline_cancels_transcription_before_late_partial():
    coordinator = ProtocolCoordinator(
        ["late partial"],
        operation_delays={"chunk": 0.08},
    )
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=0.001,
        asr_max_session_seconds=0.03,
        asr_idle_timeout_seconds=1.0,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 16)
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert event["code"] == "session_timeout"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()

    assert "chunk" not in coordinator.completed_operations
    assert coordinator.abort_count >= 1


def test_stateful_post_inference_work_cannot_emit_partial_after_session_deadline():
    coordinator = ProtocolCoordinator(
        ["late partial"],
        operation_delays={"add": 0.08},
    )
    _override_protocol(
        coordinator,
        asr_max_session_seconds=0.05,
        asr_idle_timeout_seconds=1.0,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 16)
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert event["code"] == "session_timeout"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()

    assert coordinator.abort_count >= 1


def test_chunked_post_inference_cleanup_cannot_emit_partial_after_session_deadline(
    monkeypatch,
):
    coordinator = ProtocolCoordinator(
        ["late partial"],
        operation_delays={"chunk": 0.01},
    )
    original_remove_file = asr_api.remove_file

    def delayed_remove_file(path):
        import time

        original_remove_file(path)
        time.sleep(0.08)

    monkeypatch.setattr(asr_api, "remove_file", delayed_remove_file)
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=0.001,
        asr_max_session_seconds=0.05,
        asr_idle_timeout_seconds=1.0,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 16)
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert event["code"] == "session_timeout"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()

    assert coordinator.abort_count >= 1


def test_transcript_send_is_cancelled_when_it_crosses_session_deadline():
    class DelayedWebSocket:
        def __init__(self):
            self.sent = []
            self.close_code = None

        async def send_json(self, payload):
            if payload["type"] != "error":
                await asyncio.sleep(0.06)
            self.sent.append(payload)

        async def close(self, code):
            self.close_code = code

    async def scenario():
        websocket = DelayedWebSocket()
        coordinator = ProtocolCoordinator()
        controller = asr_api.StreamingSessionController(
            websocket,
            _protocol_settings(asr_max_session_seconds=0.03),
            coordinator,
        )
        controller.session_id = "protocol-session"
        event = controller.transcript.new_event("partial", "must not be sent")

        with pytest.raises(asr_api._StreamClosed):
            await controller._send_event(event)
        return websocket, coordinator

    websocket, coordinator = asyncio.run(scenario())

    assert websocket.sent == [
        {
            "type": "error",
            "code": "session_timeout",
            "message": "Maximum session duration exceeded",
            "sequence": 2,
        }
    ]
    assert websocket.close_code == 1008
    assert coordinator.abort_count == 1


def test_stream_v2_maps_segment_reset_failure_to_stable_error():
    coordinator = ProtocolCoordinator(reset_error=RuntimeError("reset failed"))
    _override_protocol(coordinator)
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_json({"type": "segment"})
            event = websocket.receive_json()
            assert event["code"] == "inference_error"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1011)
    finally:
        _clear_asr_dependency_overrides()


def test_chunked_stream_enforces_session_timeout():
    coordinator = ProtocolCoordinator([])
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_max_session_seconds=0.01,
        asr_idle_timeout_seconds=1.0,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            event = websocket.receive_json()
            assert event["code"] == "session_timeout"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1008)
    finally:
        _clear_asr_dependency_overrides()


@pytest.mark.parametrize("raw_command", ["not json", "[]", '{"type":"bogus"}'])
def test_chunked_stream_rejects_every_invalid_json_command(raw_command):
    coordinator = ProtocolCoordinator([])
    _override_protocol(coordinator, asr_stream_mode="chunked")
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_text(raw_command)
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert event["code"] == "invalid_message"
            assert event["sequence"] == 2
            _assert_closed(websocket, 1003)
    finally:
        _clear_asr_dependency_overrides()


def test_chunked_stream_preserves_repeated_independent_chunks():
    coordinator = ProtocolCoordinator(["你好", "你好"])
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=0.01,
        asr_stable_commit_enabled=False,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 160)
            assert websocket.receive_json() == {
                "type": "partial",
                "text": "你好",
                "sequence": 2,
            }
            websocket.send_bytes(b"\x00\x00" * 160)
            assert websocket.receive_json() == {
                "type": "partial",
                "text": "你好你好",
                "sequence": 3,
            }
            websocket.send_json({"type": "end"})
            assert websocket.receive_json() == {
                "type": "final",
                "text": "你好你好",
                "sequence": 4,
            }
            _assert_closed(websocket, 1000)
    finally:
        _clear_asr_dependency_overrides()


def test_chunked_explicit_segments_flush_each_sub_chunk_without_loss_or_duplication():
    class ExactChunkCoordinator(ProtocolCoordinator):
        def __init__(self):
            super().__init__()
            self.chunk_pcm = []
            self.texts = iter(["甲", "乙", "丙"])

        async def transcribe_stream_chunk(
            self,
            _session_id,
            audio_path,
            _language,
            _audio_seconds,
        ):
            with wave.open(audio_path, "rb") as wav_file:
                self.chunk_pcm.append(wav_file.readframes(wav_file.getnframes()))
            return TranscriptionResult(next(self.texts), "zh")

    first = _pcm_s16le_samples(101, 73)
    second = _pcm_s16le_samples(202, 89)
    final = _pcm_s16le_samples(303, 107)
    coordinator = ExactChunkCoordinator()
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=1.0,
        asr_stable_commit_enabled=False,
        asr_commit_on_punctuation=False,
    )
    events = []
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            events.append(_start_stream(websocket, expect_sequence=True))

            websocket.send_bytes(first)
            websocket.send_json({"type": "segment"})
            events.extend(websocket.receive_json() for _ in range(3))

            websocket.send_json({"type": "segment"})

            websocket.send_bytes(second)
            websocket.send_json({"type": "segment"})
            events.extend(websocket.receive_json() for _ in range(3))

            websocket.send_bytes(final)
            websocket.send_json({"type": "end"})
            events.extend(websocket.receive_json() for _ in range(2))
            _assert_closed(websocket, 1000)
    finally:
        _clear_asr_dependency_overrides()

    assert coordinator.chunk_pcm == [first, second, final]
    assert [(event["type"], event.get("text", "")) for event in events] == [
        ("ready", ""),
        ("partial", "甲"),
        ("sentence_final", "甲"),
        ("partial", ""),
        ("partial", "乙"),
        ("sentence_final", "乙"),
        ("partial", ""),
        ("partial", "丙"),
        ("final", "丙"),
    ]
    assert [event["sequence"] for event in events] == list(range(1, 10))
    confirmed = "".join(
        event["text"] for event in events if event["type"] == "sentence_final"
    )
    final_tail = next(event["text"] for event in events if event["type"] == "final")
    assert confirmed + final_tail == "甲乙丙"


def test_chunked_stream_honors_legacy_immediate_punctuation_when_stable_enabled():
    coordinator = ProtocolCoordinator(["你好。"])
    _override_protocol(
        coordinator,
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=0.01,
        asr_stable_commit_enabled=True,
        asr_commit_on_punctuation=True,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 160)
            assert websocket.receive_json() == {
                "type": "sentence_final",
                "text": "你好。",
                "sequence": 2,
            }
            assert websocket.receive_json() == {
                "type": "partial",
                "text": "",
                "sequence": 3,
            }
    finally:
        _clear_asr_dependency_overrides()


@pytest.mark.parametrize(
    ("stream_mode", "stable_enabled"),
    [("stateful", False), ("chunked", True)],
)
def test_legacy_protocol_commits_every_complete_sentence_in_one_update(
    stream_mode,
    stable_enabled,
):
    coordinator = ProtocolCoordinator(["第一句。第二句。"])
    _override_protocol(
        coordinator,
        asr_stream_mode=stream_mode,
        asr_stream_chunk_seconds=0.001,
        asr_stable_commit_enabled=stable_enabled,
        asr_commit_on_punctuation=True,
    )
    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket, expect_sequence=True)
            websocket.send_bytes(b"\x00\x00" * 16)
            if stream_mode == "stateful":
                assert websocket.receive_json() == {
                    "type": "partial",
                    "text": "第一句。第二句。",
                    "sequence": 2,
                }
            else:
                assert websocket.receive_json() == {
                    "type": "sentence_final",
                    "text": "第一句。",
                    "sequence": 2,
                }
                assert websocket.receive_json() == {
                    "type": "sentence_final",
                    "text": "第二句。",
                    "sequence": 3,
                }
                assert websocket.receive_json() == {
                    "type": "partial",
                    "text": "",
                    "sequence": 4,
                }
    finally:
        _clear_asr_dependency_overrides()


def test_qwen_vllm_backend_can_be_selected():
    from app.asr import QwenVLLMASRTranscriber, create_asr_transcriber
    from app.config import Settings

    transcriber = create_asr_transcriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )

    assert isinstance(transcriber, QwenVLLMASRTranscriber)


def test_mock_backend_supports_stateful_stream_lifecycle():
    from app.asr import create_asr_transcriber
    from app.config import Settings

    session = create_asr_transcriber(
        Settings(_env_file=None, asr_backend="mock", asr_stream_mode="stateful")
    ).create_streaming_session("zh")

    update = session.add_pcm_s16le(b"\x00\x00" * 160, 16000)
    segment = session.finish_segment()
    continued = session.add_pcm_s16le(b"\x00\x00" * 160, 16000)
    final = session.finish()

    assert update.processed_samples == 160
    assert update.model_updated is True
    assert segment.segment_finished is True
    assert continued.processed_samples == 160
    assert final.text == continued.text


def test_qwen_vllm_streaming_session_feeds_pcm_and_finishes(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    calls = []

    class FakeState:
        text = ""
        language = "zh"

    class FakeModel:
        def init_streaming_state(self, **kwargs):
            calls.append(("init", kwargs))
            return FakeState()

        def streaming_transcribe(self, pcm, state):
            calls.append(("stream", len(pcm)))
            state.text = "可以到店"
            state.language = "zh"
            return state

        def finish_streaming_transcribe(self, state):
            calls.append(("finish", None))
            state.text = "可以到店使用"
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **kwargs):
            calls.append(("load", kwargs))
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))
    monkeypatch.delenv("ASR_STREAM_CHUNK_SECONDS", raising=False)

    transcriber = QwenVLLMASRTranscriber(
        Settings(
            asr_backend="qwen_vllm",
            api_key=TEST_ONLY_LONG_API_KEY,
            asr_max_frame_bytes=32000,
            asr_ws_max_queue=1,
        )
    )
    session = transcriber.create_streaming_session(language="zh")
    update = session.add_pcm_s16le((1000).to_bytes(2, "little", signed=True) * 16000, sample_rate=16000)
    final = session.finish()

    assert update.text == "可以到店"
    assert final.text == "可以到店使用"
    assert calls[0] == (
        "load",
        {
            "model": "/models/Qwen3-ASR-1.7B-hf",
            "gpu_memory_utilization": 0.8,
            "max_model_len": 65536,
            "max_new_tokens": 32,
        },
    )
    assert calls[1] == (
        "init",
        {
            "language": "Chinese",
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
            "chunk_size_sec": 2.0,
        },
    )


def test_qwen_vllm_warmup_loads_model_once(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    constructor_count = 0

    class FakeModel:
        pass

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            nonlocal constructor_count
            constructor_count += 1
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))

    transcriber = QwenVLLMASRTranscriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )
    transcriber.warmup()
    transcriber.warmup()

    assert constructor_count == 1


def test_qwen_vllm_stateful_warmup_fails_fast_without_vad_asset(
    monkeypatch, tmp_path
):
    from app.asr import QwenVLLMASRTranscriber
    from app.asr_vad import VADRuntimeError
    from app.config import Settings

    transcriber = QwenVLLMASRTranscriber(
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            api_key=TEST_ONLY_LONG_API_KEY,
            asr_vad_model_path=str(tmp_path / "missing.onnx"),
        )
    )
    monkeypatch.setattr(transcriber, "_load", lambda: None)
    monkeypatch.setattr(transcriber, "_validate_qwen_runtime_contract", lambda: None)

    with pytest.raises(VADRuntimeError, match="asset is missing"):
        transcriber.warmup()


def test_qwen_vllm_stateful_warmup_performs_non_silent_streaming_decode(
    monkeypatch,
):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    calls = []

    class FakeState:
        def __init__(self, chunk_size_samples):
            self.text = ""
            self.language = "zh"
            self.chunk_id = 0
            self.chunk_size_samples = chunk_size_samples
            self.buffered = 0

    class FakeModel:
        def init_streaming_state(self, **kwargs):
            calls.append(("init", kwargs))
            return FakeState(round(kwargs["chunk_size_sec"] * 16000))

        def streaming_transcribe(self, pcm, state):
            calls.append(("stream", len(pcm), int(max(pcm)), int(min(pcm))))
            state.buffered += len(pcm)
            while state.buffered >= state.chunk_size_samples:
                state.buffered -= state.chunk_size_samples
                state.chunk_id += 1
                state.text = "warm"
            return state

        def finish_streaming_transcribe(self, state):
            calls.append(("finish", state.buffered))
            if state.buffered:
                state.buffered = 0
                state.chunk_id += 1
                state.text = "warm"
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel),
    )
    monkeypatch.setattr(
        "app.asr.create_vad_endpoint_detector", lambda _settings: object()
    )
    transcriber = QwenVLLMASRTranscriber(
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            api_key=TEST_ONLY_LONG_API_KEY,
            asr_stream_chunk_seconds=0.5,
        )
    )
    monkeypatch.setattr(transcriber, "_validate_qwen_runtime_contract", lambda: None)

    transcriber.warmup()

    stream_calls = [call for call in calls if call[0] == "stream"]
    assert sum(call[1] for call in stream_calls) >= 16000
    assert all(call[1] <= 8000 for call in stream_calls)
    assert all(call[2] > 0 for call in stream_calls)
    assert all(call[3] < 0 for call in stream_calls)
    assert any(call[0] == "finish" for call in calls)
    assert transcriber.streaming_warmup_complete is True


def test_stateful_segment_reset_reinitializes_official_state(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    states = []

    class FakeState:
        def __init__(self):
            self.text = ""
            self.language = "zh"

    class FakeModel:
        def init_streaming_state(self, **_kwargs):
            state = FakeState()
            states.append(state)
            return state

        def streaming_transcribe(self, _pcm, state):
            state.text = "hello" if state is states[0] else " world"
            return state

        def finish_streaming_transcribe(self, state):
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))

    transcriber = QwenVLLMASRTranscriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )
    session = transcriber.create_streaming_session(language="zh")
    first = session.add_pcm_s16le(b"\x00\x00", 16000)
    segment = session.reset_segment()
    result = session.add_pcm_s16le(b"\x00\x00", 16000)

    assert first.segment_id == 0
    assert segment.segment_id == 0
    assert result.segment_id == 1
    assert result.segment_text == " world"
    assert len(states) == 2


def test_stateful_segment_finish_flushes_buffer_before_reinitializing(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    states = []
    finish_calls = []

    class FakeState:
        def __init__(self):
            self.text = ""
            self.language = "zh"
            self.buffered_samples = 0
            self.chunk_id = 0

    class FakeModel:
        def init_streaming_state(self, **_kwargs):
            state = FakeState()
            states.append(state)
            return state

        def streaming_transcribe(self, pcm, state):
            state.buffered_samples += len(pcm)
            return state

        def finish_streaming_transcribe(self, state):
            finish_calls.append(state)
            if state.buffered_samples:
                state.text = "尾音"
                state.chunk_id += 1
                state.buffered_samples = 0
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))
    session = QwenVLLMASRTranscriber(
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            api_key=TEST_ONLY_LONG_API_KEY,
            asr_stream_chunk_seconds=1.0,
            asr_max_frame_bytes=28800,
            asr_ws_max_queue=1,
        )
    ).create_streaming_session("zh")

    buffered = session.add_pcm_s16le(b"\x00\x00" * 14400, 16000)
    segment = session.finish_segment()
    continued = session.add_pcm_s16le(b"\x00\x00" * 1600, 16000)

    assert buffered.model_updated is False
    assert buffered.processed_samples == 0
    assert segment.segment_text == "尾音"
    assert segment.segment_id == 0
    assert segment.model_updated is True
    assert segment.processed_samples == 14400
    assert segment.segment_finished is True
    assert continued.segment_text == ""
    assert continued.segment_id == 1
    assert len(states) == 2
    assert finish_calls == [states[0]]


def test_stateful_rollover_bounds_official_state_and_preserves_exact_text(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    states = []
    finish_calls = []
    segment_texts = [("甲", "甲乙"), ("丙", "丙丁"), ("戊", "戊")]

    class FakeState:
        def __init__(self):
            self.text = ""
            self.language = "zh"
            self.buffered_samples = 0
            self.chunk_id = 0
            self.total_samples = 0

    class FakeModel:
        def init_streaming_state(self, **_kwargs):
            state = FakeState()
            states.append(state)
            return state

        def streaming_transcribe(self, pcm, state):
            state.buffered_samples += len(pcm)
            state.total_samples += len(pcm)
            if state.buffered_samples >= 1600:
                state.buffered_samples = 0
                state.chunk_id += 1
                state_index = states.index(state)
                state.text = segment_texts[state_index][min(state.chunk_id, 2) - 1]
            return state

        def finish_streaming_transcribe(self, state):
            finish_calls.append(state)
            if state.buffered_samples:
                state.chunk_id += 1
                state.buffered_samples = 0
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))
    session = QwenVLLMASRTranscriber(
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            api_key=TEST_ONLY_LONG_API_KEY,
            asr_stream_chunk_seconds=0.1,
            asr_stream_rollover_seconds=120.0,
            asr_max_utterance_seconds=0.25,
            asr_max_frame_bytes=3200,
        )
    ).create_streaming_session("zh")

    results = [session.add_pcm_s16le(b"\x00\x00" * 1600, 16000) for _ in range(6)]

    assert [result.segment_text for result in results if result.segment_finished] == [
        "甲乙",
        "丙丁",
    ]
    assert [result.segment_id for result in results if result.segment_finished] == [0, 1]
    assert len(finish_calls) == 2
    assert len(states) == 3
    assert [state.total_samples for state in states] == [4000, 4000, 1600]
    assert sum(state.total_samples for state in states) == 6 * 1600
    assert results[2].continuation is not None
    assert results[2].continuation.segment_id == 1
    assert session.add_pcm_s16le(b"\x00\x00" * 1600, 16000).segment_text == "戊"
    assert session.finish().segment_text == "戊"


def test_stateful_invariant_watchdog_aborts_instead_of_normal_rollover(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    stream_calls = 0

    class FakeState:
        text = ""
        language = "zh"
        chunk_id = 0
        chunk_size_samples = 1600

    class FakeModel:
        def init_streaming_state(self, **_kwargs):
            return FakeState()

        def streaming_transcribe(self, _pcm, state):
            nonlocal stream_calls
            stream_calls += 1
            return state

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel),
    )
    session = QwenVLLMASRTranscriber(
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            api_key=TEST_ONLY_LONG_API_KEY,
        )
    ).create_streaming_session("zh")
    session._max_utterance_samples = 100000
    session._watchdog_samples = 1000

    with pytest.raises(RuntimeError, match="invariant watchdog"):
        session.add_pcm_s16le(b"\x00\x00" * 1001, 16000)

    assert stream_calls == 0


def test_stateful_abort_releases_state(monkeypatch):
    import sys
    import types

    import pytest

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    class FakeState:
        text = ""
        language = "zh"

    class FakeModel:
        def init_streaming_state(self, **_kwargs):
            return FakeState()

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **_kwargs):
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))

    transcriber = QwenVLLMASRTranscriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )
    session = transcriber.create_streaming_session(language="zh")
    session.abort()

    with pytest.raises(RuntimeError, match="streaming session is closed"):
        session.add_pcm_s16le(b"\x00\x00", 16000)


def test_qwen_vllm_file_transcribe_normalizes_language_code(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    calls = []

    class FakeResult:
        text = "hello"
        language = "English"

    class FakeModel:
        def transcribe(self, **kwargs):
            calls.append(("transcribe", kwargs))
            return [FakeResult()]

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **kwargs):
            calls.append(("load", kwargs))
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))

    transcriber = QwenVLLMASRTranscriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )
    result = transcriber.transcribe("sample.wav", language="en")

    assert result.text == "hello"
    assert result.language == "English"
    assert calls[1] == ("transcribe", {"audio": "sample.wav", "language": "English"})

def test_qwen_vllm_file_transcribe_normalizes_regional_language_code(monkeypatch):
    import sys
    import types

    from app.asr import QwenVLLMASRTranscriber
    from app.config import Settings

    calls = []

    class FakeResult:
        text = "hello"
        language = "English"

    class FakeModel:
        def transcribe(self, **kwargs):
            calls.append(("transcribe", kwargs))
            return [FakeResult()]

    class FakeQwen3ASRModel:
        @classmethod
        def LLM(cls, **kwargs):
            calls.append(("load", kwargs))
            return FakeModel()

    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQwen3ASRModel))

    transcriber = QwenVLLMASRTranscriber(
        Settings(asr_backend="qwen_vllm", api_key=TEST_ONLY_LONG_API_KEY)
    )
    transcriber.transcribe("sample.wav", language="en-US")

    assert calls[1] == ("transcribe", {"audio": "sample.wav", "language": "English"})
