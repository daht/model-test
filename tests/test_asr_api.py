import os

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"
os.environ["ASR_BACKEND"] = "mock"
os.environ["ASR_STREAM_CHUNK_SECONDS"] = "0.01"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.asr_api import app  # noqa: E402


client = TestClient(app)


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
    assert body["start_message"]["type"] == "start"
    assert body["end_message"] == {"type": "end"}


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


def test_stream_returns_partial_and_final_transcripts():
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

        websocket.send_bytes(b"\x00\x00" * 320)
        partial = websocket.receive_json()
        assert partial["type"] == "partial"
        assert partial["text"] == "[mock asr zh] stream.wav"

        websocket.send_json({"type": "end"})
        final = websocket.receive_json()
        assert final["type"] == "final"
        assert final["text"] == "[mock asr zh] stream.wav"
