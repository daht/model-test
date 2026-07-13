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
    assert settings.asr_max_utterance_seconds == 30.0
    assert settings.asr_state_watchdog_seconds == 120.0
    assert settings.asr_vad_pre_roll_ms == 200
    assert settings.asr_vad_frame_samples == 512


def test_protocol_version_accepts_string_from_environment(monkeypatch):
    monkeypatch.setenv("ASR_PROTOCOL_VERSION", "2")

    settings = Settings(_env_file=None)

    assert settings.asr_protocol_version == 2


@pytest.mark.parametrize("version", ["1", "3"])
def test_protocol_version_rejects_other_environment_values(monkeypatch, version):
    monkeypatch.setenv("ASR_PROTOCOL_VERSION", version)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


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
        ("asr_max_utterance_seconds", 0),
        ("asr_state_watchdog_seconds", 0),
        ("asr_vad_min_speech_ms", 0),
        ("asr_vad_min_silence_ms", 0),
        ("asr_vad_hangover_ms", -1),
        ("asr_max_undecoded_age_seconds", 0),
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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("asr_vllm_gpu_memory_utilization", 0),
        ("asr_vllm_gpu_memory_utilization", 1),
        ("asr_vllm_gpu_memory_utilization", 1.5),
        ("asr_vllm_max_new_tokens", 0),
        ("asr_stream_unfixed_chunk_num", -1),
        ("asr_stream_unfixed_token_num", -1),
        ("asr_vad_rms_threshold", -1),
        ("asr_stream_rollover_seconds", 0),
    ],
)
def test_asr_model_settings_reject_invalid_numeric_bounds(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_qwen_transformers_backend_rejects_stateful_streaming():
    with pytest.raises(ValidationError, match="qwen.*stateful"):
        Settings(_env_file=None, asr_backend="qwen", asr_stream_mode="stateful")


@pytest.mark.parametrize(
    ("backend", "stream_mode"),
    [
        ("mock", "stateful"),
        ("qwen", "chunked"),
        ("qwen_vllm", "stateful"),
    ],
)
def test_supported_asr_backend_stream_mode_pairs_pass(backend, stream_mode):
    settings = Settings(
        _env_file=None,
        asr_backend=backend,
        asr_stream_mode=stream_mode,
    )

    assert (settings.asr_backend, settings.asr_stream_mode) == (backend, stream_mode)


def test_rollover_must_exceed_model_chunk_and_transport_frame():
    with pytest.raises(ValidationError, match="rollover"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_stream_chunk_seconds=2.0,
            asr_stream_rollover_seconds=1.0,
        )


def test_normal_utterance_limit_must_exceed_model_chunk_and_transport_frame():
    with pytest.raises(ValidationError, match="normal utterance"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_stream_chunk_seconds=2.0,
            asr_max_utterance_seconds=1.0,
        )

    with pytest.raises(ValidationError, match="normal utterance"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_stream_chunk_seconds=0.1,
            asr_max_utterance_seconds=0.4,
            asr_max_frame_bytes=16000,
        )

    with pytest.raises(ValidationError, match="VAD frame"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_max_frame_bytes=512,
        )

    with pytest.raises(ValidationError, match="rollover"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_stream_chunk_seconds=0.1,
            asr_stream_rollover_seconds=0.4,
            asr_max_frame_bytes=16000,
        )


def test_chunked_stream_does_not_apply_stateful_rollover_relationships():
    settings = Settings(
        _env_file=None,
        asr_backend="qwen",
        asr_stream_mode="chunked",
        asr_stream_chunk_seconds=2.0,
        asr_stream_rollover_seconds=1.0,
    )

    assert settings.asr_stream_rollover_seconds == 1.0


def test_stateful_vad_thresholds_and_durations_are_cross_validated():
    with pytest.raises(ValidationError, match="onset.*offset"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_vad_onset_threshold=0.3,
            asr_vad_offset_threshold=0.4,
        )

    with pytest.raises(ValidationError, match="hangover"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_vad_min_silence_ms=100,
            asr_vad_hangover_ms=101,
        )

    with pytest.raises(ValidationError, match="watchdog"):
        Settings(
            _env_file=None,
            asr_backend="qwen_vllm",
            asr_stream_mode="stateful",
            asr_max_utterance_seconds=30,
            asr_state_watchdog_seconds=30,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("asr_vad_onset_threshold", 0),
        ("asr_vad_onset_threshold", 1),
        ("asr_vad_offset_threshold", 0),
        ("asr_vad_offset_threshold", 1),
        ("asr_vad_pre_roll_ms", 199),
        ("asr_vad_pre_roll_ms", 201),
        ("asr_vad_frame_samples", 256),
        ("asr_vad_frame_samples", 1024),
        ("asr_vad_onnx_intra_threads", 0),
        ("asr_vad_onnx_inter_threads", 0),
    ],
)
def test_stateful_vad_settings_reject_out_of_contract_values(name, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: value})


def test_silero_asset_identity_is_pinned_and_checksum_is_strict():
    settings = Settings(_env_file=None)

    assert settings.asr_vad_model_version == "6.2.1"
    assert len(settings.asr_vad_model_sha256) == 64
    assert set(settings.asr_vad_model_sha256) <= set("0123456789abcdef")

    with pytest.raises(ValidationError, match="SHA256"):
        Settings(_env_file=None, asr_vad_model_sha256="not-a-checksum")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, asr_vad_model_version="latest")
    with pytest.raises(ValidationError, match="pinned"):
        Settings(_env_file=None, asr_vad_model_sha256="0" * 64)
