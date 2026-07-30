import asyncio
import json
import os
import struct
from threading import BoundedSemaphore, Event

from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key"
os.environ["ASR_BACKEND"] = "mock"
os.environ["TTS_BACKEND"] = "mock"
os.environ["TTS_MODEL_NAME"] = "CosyVoice"
os.environ["TTS_MAX_TEXT_CHARS"] = "12"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.tts_api import _stream_pcm_in_thread, app  # noqa: E402


client = TestClient(app)


def _connect(api_key: str = "test-key"):
    return client.websocket_connect(
        "/v1/tts/stream",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _start_task(websocket, transport: str = "hex", expect_started: bool = True) -> dict:
    connected = websocket.receive_json()
    assert connected["event"] == "connected_success"
    assert connected["base_resp"] == {"status_code": 0, "status_msg": "success"}

    websocket.send_json(
        {
            "event": "task_start",
            "model": "CosyVoice",
            "voice_setting": {"voice_id": "default"},
            "audio_setting": {
                "sample_rate": 24000,
                "format": "pcm",
                "channel": 1,
            },
            "stream_options": {"audio_transport": transport},
        }
    )
    response = websocket.receive_json()
    if expect_started:
        assert response["event"] == "task_started"
        assert response["base_resp"] == {"status_code": 0, "status_msg": "success"}
    return response


def test_tts_health_reports_model_name():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "CosyVoice"
    assert response.json()["backend"] == "mock"
    assert response.json()["sample_rate"] == 24000


def test_tts_capacity_requires_api_key():
    response = client.get("/v1/tts/capacity")

    assert response.status_code == 401


def test_tts_capacity_reports_selected_backend():
    response = client.get("/v1/tts/capacity", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "CosyVoice"
    assert body["backend"] == "mock"
    assert body["sample_rate"] == 24000
    assert body["supports_microbatch"] is False
    assert body["queue_depth"] is None


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
    with _connect(api_key="bad-key") as websocket:
        failed = websocket.receive_json()

        assert failed["event"] == "task_failed"
        assert failed["base_resp"]["status_code"] == 1001
        assert failed["base_resp"]["status_msg"] == "Invalid or missing API key"


def test_tts_stream_rejects_when_active_capacity_is_exhausted(monkeypatch):
    slots = BoundedSemaphore(1)
    assert slots.acquire(blocking=False)
    monkeypatch.setattr("app.tts_api.tts_stream_slots", slots)

    with _connect() as websocket:
        failed = _start_task(websocket, expect_started=False)
        assert failed["event"] == "task_failed"
        assert failed["base_resp"] == {
            "status_code": 1013,
            "status_msg": "TTS is at capacity",
        }


def test_tts_stream_hex_mode_returns_multiple_pcm_chunks():
    with _connect() as websocket:
        _start_task(websocket, transport="hex")
        websocket.send_json({"event": "task_continue", "text": "hello"})
        websocket.send_json({"event": "task_finish"})

        chunks = []
        sequences = []
        while True:
            message = websocket.receive_json()
            if message["event"] == "task_finished":
                finished = message
                break
            assert message["event"] == "task_continued"
            chunks.append(bytes.fromhex(message["data"]["audio"]))
            sequences.append(message["extra_info"]["chunk_sequence"])

        assert len(chunks) > 1
        assert all(chunks)
        assert sequences == list(range(len(chunks)))
        assert finished["extra_info"]["chunks"] == len(chunks)
        assert finished["extra_info"]["total_samples"] == sum(len(chunk) for chunk in chunks) // 2


def test_tts_stream_binary_mode_uses_tts1_header_and_contiguous_offsets():
    with _connect() as websocket:
        _start_task(websocket, transport="binary")
        websocket.send_json({"event": "task_continue", "text": "hello"})
        websocket.send_json({"event": "task_finish"})

        chunks = []
        expected_offset = 0
        while True:
            message = websocket.receive()
            if message.get("text") is not None:
                finished = json.loads(message["text"])
                assert finished["event"] == "task_finished"
                break
            payload = message["bytes"]
            magic, sequence, sample_offset = struct.unpack("<4sIQ", payload[:16])
            pcm = payload[16:]
            assert magic == b"TTS1"
            assert sequence == len(chunks)
            assert sample_offset == expected_offset
            assert len(pcm) % 2 == 0
            chunks.append(pcm)
            expected_offset += len(pcm) // 2

        assert len(chunks) > 1
        assert finished["extra_info"]["total_samples"] == expected_offset


def test_tts_stream_rejects_blank_text_with_task_failed():
    with _connect() as websocket:
        _start_task(websocket)
        websocket.send_json({"event": "task_continue", "text": "   "})

        failed = websocket.receive_json()
        assert failed["event"] == "task_failed"
        assert failed["base_resp"]["status_code"] == 1004
        assert failed["base_resp"]["status_msg"] == "text cannot be blank"


def test_tts_stream_rejects_unsupported_audio_settings():
    with _connect() as websocket:
        connected = websocket.receive_json()
        assert connected["event"] == "connected_success"
        websocket.send_json(
            {
                "event": "task_start",
                "model": "CosyVoice",
                "voice_setting": {"voice_id": "default"},
                "audio_setting": {"sample_rate": 32000, "format": "mp3", "channel": 2},
            }
        )

        failed = websocket.receive_json()
        assert failed["event"] == "task_failed"
        assert failed["base_resp"]["status_code"] == 1003


def test_thread_bridge_releases_first_chunk_without_waiting_for_second():
    allow_second = Event()

    class ControlledSynthesizer:
        def stream_pcm(self, text, voice=None):
            yield b"\x01\x00"
            assert allow_second.wait(timeout=1)
            yield b"\x02\x00"

    async def consume():
        stream = _stream_pcm_in_thread(ControlledSynthesizer(), "hello", "default")
        first = await anext(stream)
        assert first == b"\x01\x00"
        allow_second.set()
        assert [chunk async for chunk in stream] == [b"\x02\x00"]

    asyncio.run(consume())
