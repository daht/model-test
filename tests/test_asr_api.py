import os

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"
os.environ["ASR_BACKEND"] = "mock"
os.environ["ASR_STREAM_CHUNK_SECONDS"] = "0.01"
os.environ["ASR_COMMIT_ON_PUNCTUATION"] = "false"
os.environ["ASR_VAD_SILENCE_SECONDS"] = "1.5"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

import app.asr_api as asr_api  # noqa: E402
from app.asr_api import app  # noqa: E402


client = TestClient(app)


def _start_stream(websocket):
    websocket.send_json(
        {
            "type": "start",
            "api_key": "test-key",
            "language": "zh",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        }
    )
    assert websocket.receive_json() == {"type": "ready"}


def _send_transcribable_chunk(websocket):
    websocket.send_bytes(b"\x00\x00" * 160)


def _pcm_s16le_samples(value: int, sample_count: int) -> bytes:
    return int(value).to_bytes(2, byteorder="little", signed=True) * sample_count


class FakeStreamingSession:
    def __init__(self, updates, final_text="hello world"):
        self.updates = iter(updates)
        self.finished = False
        self.final_text = final_text

    def add_pcm_s16le(self, _pcm_bytes, _sample_rate):
        from app.asr import StreamingTranscriptionResult

        text = next(self.updates)
        return StreamingTranscriptionResult(text=text, language="zh")

    def finish(self):
        from app.asr import StreamingTranscriptionResult

        self.finished = True
        return StreamingTranscriptionResult(text=self.final_text, language="zh")


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


def test_asr_health_reports_model_name():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "Qwen3-ASR-1.7B"
    assert response.json()["backend"] == "mock"


def test_transcribe_rejects_missing_api_key():
    response = client.post(
        "/v1/transcribe",
        files={"file": ("sample.wav", b"fake wav", "audio/wav")},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_transcribe_accepts_audio_upload_with_language_hint():
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": "test-key"},
        data={"language": "en"},
        files={"file": ("sample.wav", b"fake wav", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "[mock asr en] sample.wav",
        "language": "en",
        "model": "Qwen3-ASR-1.7B",
    }


def test_transcribe_rejects_unsupported_extension():
    response = client.post(
        "/v1/transcribe",
        headers={"X-API-Key": "test-key"},
        files={"file": ("sample.txt", b"not audio", "text/plain")},
    )

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
    assert {"type": "sentence_final", "text": "..."} in body["server_messages"]


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
    monkeypatch.setenv("ASR_VLLM_MAX_NEW_TOKENS", "32")

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
    assert body["audio_format"]["stateful"]["vllm_max_new_tokens"] == 32


def test_qwen_vllm_backend_can_be_selected():
    from app.asr import QwenVLLMASRTranscriber, create_asr_transcriber
    from app.config import Settings

    transcriber = create_asr_transcriber(Settings(asr_backend="qwen_vllm"))

    assert isinstance(transcriber, QwenVLLMASRTranscriber)


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

    transcriber = QwenVLLMASRTranscriber(Settings(asr_backend="qwen_vllm"))
    session = transcriber.create_streaming_session(language="zh")
    update = session.add_pcm_s16le((1000).to_bytes(2, "little", signed=True) * 16000, sample_rate=16000)
    final = session.finish()

    assert update.text == "可以到店"
    assert final.text == "可以到店使用"
    assert calls[0][0] == "load"
    assert calls[1] == (
        "init",
        {
            "language": "Chinese",
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
            "chunk_size_sec": 2.0,
        },
    )


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

    transcriber = QwenVLLMASRTranscriber(Settings(asr_backend="qwen_vllm"))
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

    transcriber = QwenVLLMASRTranscriber(Settings(asr_backend="qwen_vllm"))
    transcriber.transcribe("sample.wav", language="en-US")

    assert calls[1] == ("transcribe", {"audio": "sample.wav", "language": "English"})


def test_stream_rejects_bad_api_key():
    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "api_key": "bad-key",
                "language": "zh",
                "sample_rate": 16000,
                "format": "pcm_s16le",
            }
        )

        assert websocket.receive_json() == {
            "type": "error",
            "message": "Invalid or missing API key",
        }


def test_stateful_stream_returns_replaceable_partial(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(asr_api, "asr_transcriber", FakeStatefulTranscriber(FakeStreamingSession(["可以到店", "可以到店使用"])))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "可以到店"}
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "可以到店使用"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_replaces_revised_partial_text(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(
        asr_api,
        "asr_transcriber",
        FakeStatefulTranscriber(FakeStreamingSession(["可以到店。", "可以到店使用"])),
    )

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "可以到店。"}
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "可以到店使用"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_vad_commit_excludes_confirmed_prefix(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(asr_api, "asr_transcriber", FakeStatefulTranscriber(FakeStreamingSession(["hello", "hello", "hello world"])))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "hello"}
            websocket.send_bytes(_pcm_s16le_samples(0, 24000))
            assert websocket.receive_json() == {"type": "sentence_final", "text": "hello"}
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": " world"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_clears_revised_empty_partial(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(asr_api, "asr_transcriber", FakeStatefulTranscriber(FakeStreamingSession(["hello", ""])))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "hello"}
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": ""}
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_finish_flushes_remaining_text(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    session = FakeStreamingSession(["hello"], final_text="hello world")
    monkeypatch.setattr(asr_api, "asr_transcriber", FakeStatefulTranscriber(session))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)
            websocket.send_bytes(_pcm_s16le_samples(1000, 160))
            assert websocket.receive_json() == {"type": "partial", "text": "hello"}
            websocket.send_json({"type": "end"})
            assert websocket.receive_json() == {"type": "partial", "text": "hello world"}
            assert websocket.receive_json() == {"type": "final", "text": "hello world"}
            assert session.finished is True
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_rejects_non_16khz_sample_rate(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(asr_api, "asr_transcriber", FakeStatefulTranscriber(FakeStreamingSession([])))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "api_key": "test-key",
                    "language": "zh",
                    "sample_rate": 8000,
                    "format": "pcm_s16le",
                }
            )
            assert websocket.receive_json() == {
                "type": "error",
                "message": "Stateful ASR streaming requires sample_rate 16000",
            }
    finally:
        asr_api.get_settings.cache_clear()


def test_stateful_stream_returns_error_when_streaming_session_rejects_language(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_STREAM_MODE", "stateful")
    monkeypatch.setattr(
        asr_api,
        "asr_transcriber",
        RaisingStatefulTranscriber(ValueError("Unsupported language: Xx")),
    )

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "api_key": "test-key",
                    "language": "xx",
                    "sample_rate": 16000,
                    "format": "pcm_s16le",
                }
            )
            assert websocket.receive_json() == {
                "type": "error",
                "message": "Unsupported language: Xx",
            }
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_returns_partial_and_final_transcripts(monkeypatch):
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "未完成文本")

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)
        partial = websocket.receive_json()
        assert partial["type"] == "partial"
        assert partial["text"] == "未完成文本"

        websocket.send_json({"type": "end"})
        final = websocket.receive_json()
        assert final["type"] == "final"
        assert final["text"] == "未完成文本"


def test_stream_keeps_punctuated_text_partial_by_default(monkeypatch):
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。")

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)

        assert websocket.receive_json() == {"type": "partial", "text": "你好。"}


def test_stream_can_commit_punctuation_when_enabled(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。")

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)

            _send_transcribable_chunk(websocket)

            assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_partial_excludes_previously_committed_text_when_punctuation_commit_enabled(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    texts = iter(["你好。", "继续说"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)

            _send_transcribable_chunk(websocket)
            assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

            _send_transcribable_chunk(websocket)
            assert websocket.receive_json() == {"type": "partial", "text": "继续说"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_handles_cumulative_asr_text_without_recommitting_when_punctuation_commit_enabled(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    texts = iter(["你好。", "你好。继续说"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)

            _send_transcribable_chunk(websocket)
            assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

            _send_transcribable_chunk(websocket)
            assert websocket.receive_json() == {"type": "partial", "text": "继续说"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_keeps_multiple_punctuated_sentences_partial_by_default(monkeypatch):
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。再见！还有半句")

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)

        assert websocket.receive_json() == {"type": "partial", "text": "你好。再见！还有半句"}


def test_stream_commits_multiple_sentences_and_partial_remainder_when_punctuation_commit_enabled(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。再见！还有半句")

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)

            _send_transcribable_chunk(websocket)

            assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}
            assert websocket.receive_json() == {"type": "sentence_final", "text": "再见！"}
            assert websocket.receive_json() == {"type": "partial", "text": "还有半句"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_end_final_only_returns_remaining_uncommitted_text_when_punctuation_commit_enabled(monkeypatch):
    asr_api.get_settings.cache_clear()
    monkeypatch.setenv("ASR_COMMIT_ON_PUNCTUATION", "true")
    texts = iter(["你好。", "最后半句"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    try:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            _start_stream(websocket)

            _send_transcribable_chunk(websocket)
            assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

            websocket.send_bytes(b"\x00\x00" * 80)
            websocket.send_json({"type": "end"})

            assert websocket.receive_json() == {"type": "partial", "text": "最后半句"}
            assert websocket.receive_json() == {"type": "final", "text": "最后半句"}
    finally:
        asr_api.get_settings.cache_clear()


def test_stream_commits_pending_text_after_silence(monkeypatch):
    texts = iter(["还没有标点。", ""])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        websocket.send_bytes(_pcm_s16le_samples(1000, 160))
        assert websocket.receive_json() == {"type": "partial", "text": "还没有标点。"}

        websocket.send_bytes(_pcm_s16le_samples(0, 24000))
        assert websocket.receive_json() == {"type": "sentence_final", "text": "还没有标点。"}

        websocket.send_json({"type": "end"})
        assert websocket.receive_json() == {"type": "final", "text": ""}


def test_stream_preserves_leading_separator_after_vad_commit_for_cumulative_text(monkeypatch):
    texts = iter(["hello", "", "hello world", ""])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        websocket.send_bytes(_pcm_s16le_samples(1000, 160))
        assert websocket.receive_json() == {"type": "partial", "text": "hello"}

        websocket.send_bytes(_pcm_s16le_samples(0, 24000))
        assert websocket.receive_json() == {"type": "sentence_final", "text": "hello"}

        websocket.send_bytes(_pcm_s16le_samples(1000, 160))
        assert websocket.receive_json() == {"type": "partial", "text": " world"}

        websocket.send_bytes(_pcm_s16le_samples(0, 24000))
        assert websocket.receive_json() == {"type": "sentence_final", "text": " world"}

        websocket.send_json({"type": "end"})
        assert websocket.receive_json() == {"type": "final", "text": ""}


def test_sentence_committer_keeps_punctuation_pending_by_default():
    committer = asr_api.SentenceCommitter()

    committed = committer.append("你好。")

    assert committed == []
    assert committer.pending_text == "你好。"


def test_sentence_committer_can_commit_punctuation_for_compatibility():
    committer = asr_api.SentenceCommitter(commit_on_punctuation=True)

    committed = committer.append("你好。")

    assert committed == ["你好。"]
    assert committer.pending_text == ""


def test_sentence_committer_does_not_split_decimal_abbreviations_or_domains():
    committer = asr_api.SentenceCommitter(commit_on_punctuation=True)

    committed = committer.append("pi is 3.14 and Dr. Smith uses e.g. example.com as a domain in the U.S.A.")

    assert committed == []
    assert committer.pending_text == "pi is 3.14 and Dr. Smith uses e.g. example.com as a domain in the U.S.A."


def test_sentence_committer_includes_closing_quotes_in_committed_sentence():
    committer = asr_api.SentenceCommitter(commit_on_punctuation=True)

    committed = committer.append("他说“你好。”后续")

    assert committed == ["他说“你好。”"]
    assert committer.pending_text == "后续"


def test_stream_segment_clears_pending_audio_buffer():
    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "api_key": "test-key",
                "language": "zh",
                "sample_rate": 16000,
                "format": "pcm_s16le",
            }
        )
        assert websocket.receive_json() == {"type": "ready"}

        websocket.send_bytes(b"\x00\x00" * 80)
        websocket.send_json({"type": "segment"})
        websocket.send_json({"type": "end"})

        final = websocket.receive_json()
        assert final["type"] == "final"
        assert final["text"] == ""
