import os

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"
os.environ["ASR_BACKEND"] = "mock"
os.environ["ASR_STREAM_CHUNK_SECONDS"] = "0.01"

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
    assert body["audio_format"]["vad_silence_seconds"] == 1.0
    assert body["start_message"]["type"] == "start"
    assert body["segment_message"] == {"type": "segment"}
    assert body["end_message"] == {"type": "end"}
    assert {"type": "sentence_final", "text": "..."} in body["server_messages"]


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


def test_stream_commits_chinese_sentence_and_does_not_send_empty_partial(monkeypatch):
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。")

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)

        assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}


def test_stream_partial_excludes_previously_committed_text(monkeypatch):
    texts = iter(["你好。", "继续说"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)
        assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

        _send_transcribable_chunk(websocket)
        assert websocket.receive_json() == {"type": "partial", "text": "继续说"}


def test_stream_handles_cumulative_asr_text_without_recommitting(monkeypatch):
    texts = iter(["你好。", "你好。继续说"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)
        assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

        _send_transcribable_chunk(websocket)
        assert websocket.receive_json() == {"type": "partial", "text": "继续说"}


def test_stream_commits_multiple_sentences_and_partial_remainder(monkeypatch):
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: "你好。再见！还有半句")

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)

        assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}
        assert websocket.receive_json() == {"type": "sentence_final", "text": "再见！"}
        assert websocket.receive_json() == {"type": "partial", "text": "还有半句"}


def test_stream_end_final_only_returns_remaining_uncommitted_text(monkeypatch):
    texts = iter(["你好。", "最后半句"])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        _send_transcribable_chunk(websocket)
        assert websocket.receive_json() == {"type": "sentence_final", "text": "你好。"}

        websocket.send_bytes(b"\x00\x00" * 80)
        websocket.send_json({"type": "end"})

        assert websocket.receive_json() == {"type": "partial", "text": "最后半句"}
        assert websocket.receive_json() == {"type": "final", "text": "最后半句"}


def test_stream_commits_pending_text_after_one_second_of_silence(monkeypatch):
    texts = iter(["还没有标点", ""])
    monkeypatch.setattr(asr_api, "_transcribe_pcm_chunk", lambda *_args: next(texts))

    with client.websocket_connect("/v1/transcribe/stream") as websocket:
        _start_stream(websocket)

        websocket.send_bytes(_pcm_s16le_samples(1000, 160))
        assert websocket.receive_json() == {"type": "partial", "text": "还没有标点"}

        websocket.send_bytes(_pcm_s16le_samples(0, 16000))
        assert websocket.receive_json() == {"type": "sentence_final", "text": "还没有标点"}

        websocket.send_json({"type": "end"})
        assert websocket.receive_json() == {"type": "final", "text": ""}


def test_sentence_committer_does_not_split_decimal_abbreviations_or_domains():
    committer = asr_api.SentenceCommitter()

    committed = committer.append("pi is 3.14 and Dr. Smith uses e.g. example.com as a domain in the U.S.A.")

    assert committed == []
    assert committer.pending_text == "pi is 3.14 and Dr. Smith uses e.g. example.com as a domain in the U.S.A."


def test_sentence_committer_includes_closing_quotes_in_committed_sentence():
    committer = asr_api.SentenceCommitter()

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
