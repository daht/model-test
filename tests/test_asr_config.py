import pytest
from pydantic import ValidationError

from app.config import Settings


def test_asr_hardening_defaults_are_conservative():
    settings = Settings(_env_file=None)

    assert settings.asr_protocol_version == 2
    assert settings.asr_eager_load is True
    assert settings.asr_file_transcribe_enabled is False
    assert settings.asr_max_active_streams == 2
    assert settings.asr_inference_queue_size == 16
    assert settings.asr_max_frame_bytes == 32000


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
    ],
)
def test_asr_hardening_settings_must_be_positive(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_pcm_frame_limit_must_be_even():
    with pytest.raises(ValidationError, match="even"):
        Settings(_env_file=None, asr_max_frame_bytes=31999)
