import os

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"
os.environ["TTS_BACKEND"] = "mock"
os.environ["TTS_MAX_TEXT_CHARS"] = "12"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.tts_api import app  # noqa: E402


client = TestClient(app)


def _start_stream(websocket, api_key: str = "test-key") -> dict:
    websocket.send_json(
        {
            "type": "start",
            "api_key": api_key,
            "voice": "default",
            "sample_rate": 24000,
            "format": "wav",
        }
    )
    return websocket.receive_json()


def test_tts_health_reports_model_name():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "CosyVoice"
    assert response.json()["backend"] == "mock"
    assert response.json()["sample_rate"] == 24000


def test_tts_rejects_missing_api_key():
    response = client.post("/v1/tts", json={"text": "hello"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_tts_returns_wav_with_api_key():
    response = client.post(
        "/v1/tts",
        headers={"X-API-Key": "test-key"},
        json={"text": "hello", "voice": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert b"WAVE" in response.content[:16]
    assert len(response.content) > 44


def test_tts_rejects_text_that_exceeds_configured_limit():
    response = client.post(
        "/v1/tts",
        headers={"X-API-Key": "test-key"},
        json={"text": "this text is too long"},
    )

    assert response.status_code == 422


def test_tts_stream_rejects_bad_api_key():
    with client.websocket_connect("/v1/tts/stream") as websocket:
        ready = _start_stream(websocket, api_key="bad-key")

        assert ready == {
            "type": "error",
            "message": "Invalid or missing API key",
        }


def test_tts_stream_text_returns_binary_audio_chunk():
    with client.websocket_connect("/v1/tts/stream") as websocket:
        assert _start_stream(websocket) == {"type": "ready"}

        websocket.send_json({"type": "text", "text": "hello"})
        chunk = websocket.receive_bytes()

        assert chunk.startswith(b"RIFF")
        assert b"WAVE" in chunk[:16]
        assert len(chunk) > 44


def test_tts_stream_rejects_blank_text():
    with client.websocket_connect("/v1/tts/stream") as websocket:
        assert _start_stream(websocket) == {"type": "ready"}

        websocket.send_json({"type": "text", "text": "   "})

        assert websocket.receive_json() == {
            "type": "error",
            "message": "text cannot be blank",
        }


def test_tts_stream_end_sends_done_and_closes():
    with client.websocket_connect("/v1/tts/stream") as websocket:
        assert _start_stream(websocket) == {"type": "ready"}

        websocket.send_json({"type": "end"})

        assert websocket.receive_json() == {"type": "done"}
