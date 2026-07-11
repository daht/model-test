from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.config import Settings


def test_asr_hardening_defaults_are_conservative():
    settings = Settings(_env_file=None)

    assert settings.asr_protocol_version == 2
    assert settings.asr_eager_load is True
    assert settings.asr_file_transcribe_enabled is False
    assert settings.asr_max_active_streams == 2
    assert settings.asr_inference_queue_size == 16
    assert settings.asr_max_frame_bytes == 16000
    assert settings.asr_ws_max_queue == 4


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("asr_max_active_streams", 0),
        ("asr_inference_queue_size", 0),
        ("asr_max_queued_audio_seconds", 0),
        ("asr_max_connection_lag_seconds", 0),
        ("asr_max_frame_bytes", 0),
        ("asr_start_timeout_seconds", 0),
        ("asr_idle_timeout_seconds", 0),
        ("asr_max_session_seconds", 0),
        ("asr_max_audio_seconds", 0),
        ("asr_stream_queue_timeout_seconds", 0),
        ("asr_stream_inference_timeout_seconds", 0),
        ("asr_stream_chunk_seconds", 0),
        ("asr_vad_silence_seconds", 0),
        ("asr_stable_commit_seconds", 0),
        ("asr_stable_commit_min_chars", 0),
        ("asr_stable_commit_min_updates", 0),
        ("asr_ws_max_queue", 0),
        ("asr_shutdown_grace_seconds", 0),
    ],
)
def test_asr_hardening_settings_must_be_positive(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_pcm_frame_limit_must_be_even():
    with pytest.raises(ValidationError, match="even"):
        Settings(_env_file=None, asr_max_frame_bytes=31999)


def test_websocket_transport_buffer_must_fit_realtime_lag_limit():
    with pytest.raises(ValidationError, match="buffered audio"):
        Settings(
            _env_file=None,
            asr_max_frame_bytes=16000,
            asr_ws_max_queue=5,
            asr_max_connection_lag_seconds=2.0,
        )

    settings = Settings(
        _env_file=None,
        asr_max_frame_bytes=32000,
        asr_ws_max_queue=2,
        asr_max_connection_lag_seconds=2.0,
    )
    assert settings.asr_ws_max_queue == 2


def test_compose_uses_asr_transport_environment_for_uvicorn_bounds():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    command = compose["services"]["qwen-asr-api"]["command"]

    assert command[command.index("--ws-max-size") + 1] == "${ASR_MAX_FRAME_BYTES:-16000}"
    assert command[command.index("--ws-max-queue") + 1] == "${ASR_WS_MAX_QUEUE:-4}"
