import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SILERO_VAD_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)
PRODUCTION_API_KEY_MIN_LENGTH = 32
PRODUCTION_API_KEY_PLACEHOLDERS = {
    "change-me",
    "replace-with-a-long-random-secret",
    "test-key",
    "your-api-key",
    "your-production-api-key",
    "<your-api-key>",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "hy-mt-rest-api"
    model_name: str = "HY-MT1.5-1.8B"
    model_id: str = Field(default="HY-MT1.5-1.8B", description="Hugging Face id or local path")
    model_backend: Literal["transformers", "mock"] = "transformers"
    model_task: Literal["causal-lm", "seq2seq-lm"] = "causal-lm"
    asr_model_name: str = "Qwen3-ASR-1.7B"
    asr_model_id: str = "/models/Qwen3-ASR-1.7B-hf"
    asr_require_model_manifest: bool = False
    asr_model_manifest_path: str | None = None
    asr_backend: Literal["qwen", "qwen_vllm", "faster_whisper", "mock"] = "qwen"
    asr_stream_mode: Literal["chunked", "stateful", "rolling"] = "chunked"
    api_key: str = Field(default="change-me", description="Required X-API-Key value")
    device: str = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "float16"
    max_new_tokens: int = 1024
    asr_device: str = "auto"
    asr_torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "bfloat16"
    asr_max_new_tokens: int = 512
    asr_max_upload_mb: int = 200
    asr_stream_chunk_seconds: float = Field(default=2.0, gt=0, le=30)
    asr_vllm_gpu_memory_utilization: float = Field(default=0.8, gt=0, lt=1)
    asr_vllm_max_model_len: int = Field(default=65536, ge=512, le=65536)
    asr_vllm_max_new_tokens: int = Field(default=32, gt=0)
    asr_faster_whisper_compute_type: Literal["float16", "int8_float16", "int8"] = "float16"
    asr_faster_whisper_batch_size: int = Field(default=4, gt=0, le=32)
    asr_faster_whisper_partial_beam_size: int = Field(default=1, gt=0, le=20)
    asr_faster_whisper_final_beam_size: int = Field(default=5, gt=0, le=20)
    asr_faster_whisper_task: Literal["transcribe"] = "transcribe"
    asr_diagnostic_logging: bool = False
    asr_slow_engine_log_seconds: float = Field(default=2.0, gt=0, le=300)
    asr_stream_unfixed_chunk_num: int = Field(default=2, ge=0)
    asr_stream_unfixed_token_num: int = Field(default=5, ge=0)
    asr_stream_rollover_seconds: float = Field(default=120.0, gt=0, le=3600)
    asr_max_utterance_seconds: float = Field(default=30.0, gt=0, le=60)
    asr_state_watchdog_seconds: float = Field(default=120.0, ge=120, le=120)
    asr_vad_silence_seconds: float = Field(default=1.5, gt=0, le=30)
    asr_vad_rms_threshold: int = Field(default=200, ge=0, le=32767)
    asr_vad_model_path: str = Field(
        default="/opt/asr-assets/silero_vad.onnx", min_length=1
    )
    asr_vad_model_version: Literal["6.2.1"] = "6.2.1"
    asr_vad_model_sha256: str = SILERO_VAD_SHA256
    asr_vad_frame_samples: int = Field(default=512, ge=512, le=512)
    asr_vad_onset_threshold: float = Field(default=0.65, gt=0, lt=1)
    asr_vad_offset_threshold: float = Field(default=0.35, gt=0, lt=1)
    asr_vad_min_speech_ms: int = Field(default=250, gt=0, le=5000)
    asr_vad_min_silence_ms: int = Field(default=800, gt=0, le=30000)
    asr_vad_hangover_ms: int = Field(default=160, ge=0, le=5000)
    asr_vad_pre_roll_ms: int = Field(default=200, ge=200, le=200)
    asr_vad_onnx_intra_threads: int = Field(default=1, ge=1, le=4)
    asr_vad_onnx_inter_threads: int = Field(default=1, ge=1, le=4)
    asr_commit_on_punctuation: bool = False
    asr_stable_commit_enabled: bool = True
    asr_stable_commit_seconds: float = Field(default=1.0, gt=0, le=30)
    asr_stable_commit_min_chars: int = Field(default=8, gt=0, le=10000)
    asr_stable_commit_min_updates: int = Field(default=2, gt=0, le=1000)
    asr_protocol_version: int = Field(default=2, ge=2, le=2)
    asr_eager_load: bool = True
    asr_file_transcribe_enabled: bool = False
    asr_max_active_streams: int = Field(default=2, gt=0, le=64)
    asr_inference_queue_size: int = Field(default=16, gt=0, le=1024)
    asr_max_queued_audio_seconds: float = Field(default=4.0, gt=0, le=120)
    asr_max_connection_lag_seconds: float = Field(default=2.0, gt=0, le=30)
    asr_max_undecoded_age_seconds: float = Field(default=4.0, gt=0, le=60)
    asr_max_frame_bytes: int = Field(default=16000, gt=0, le=1_048_576)
    asr_ws_max_queue: int = Field(default=4, gt=0, le=1024)
    asr_start_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    asr_idle_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    asr_max_session_seconds: float = Field(default=1800.0, gt=0, le=86400)
    asr_max_audio_seconds: float = Field(default=1800.0, gt=0, le=86400)
    asr_stream_queue_timeout_seconds: float = Field(default=2.0, gt=0, le=60)
    asr_stream_inference_timeout_seconds: float = Field(default=15.0, gt=0, le=300)
    asr_file_inference_timeout_seconds: float = Field(default=300.0, gt=0, le=3600)
    asr_shutdown_grace_seconds: float = Field(default=10.0, gt=0, le=300)
    asr_gateway_schedule_max_wait_ms: int = Field(default=20, ge=0, le=1000)
    asr_gateway_max_ready_jobs: int = Field(default=64, gt=0, le=4096)
    asr_gateway_max_queued_audio_seconds: float = Field(default=8.0, gt=0, le=600)
    asr_gateway_max_session_buffer_seconds: float = Field(default=4.0, gt=0, le=120)
    asr_gateway_default_update_ms: int = Field(default=1500, gt=0, le=30000)
    asr_gateway_drain_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    asr_gateway_default_backend: str = Field(default="local", min_length=1, max_length=128)
    asr_gateway_max_active_sessions: int = Field(default=2, gt=0, le=1024)
    tts_model_name: str = "CosyVoice"
    tts_backend: Literal["mock", "cosyvoice"] = "mock"
    tts_model_id: str = "/models/CosyVoice"
    tts_device: str = "auto"
    tts_sample_rate: int = 24000
    tts_max_text_chars: int = 1000
    tts_default_voice: str = "default"
    tts_cosyvoice_repo: str | None = "/opt/CosyVoice"
    trust_remote_code: bool = True

    @field_validator("asr_max_frame_bytes")
    @classmethod
    def require_even_pcm_frame_limit(cls, value: int) -> int:
        if value % 2:
            raise ValueError("asr_max_frame_bytes must be even for pcm_s16le")
        return value

    @field_validator("asr_vad_model_sha256")
    @classmethod
    def require_sha256_hex(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("asr_vad_model_sha256 must be a 64-character SHA256")
        if normalized != SILERO_VAD_SHA256:
            raise ValueError("asr_vad_model_sha256 must match the pinned model")
        return normalized

    @field_validator("asr_model_manifest_path")
    @classmethod
    def normalize_optional_manifest_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def bound_websocket_buffered_audio(self) -> "Settings":
        if self.asr_backend in {"qwen", "qwen_vllm", "faster_whisper"}:
            normalized_api_key = self.api_key.strip()
            if (
                len(normalized_api_key) < PRODUCTION_API_KEY_MIN_LENGTH
                or normalized_api_key.lower() in PRODUCTION_API_KEY_PLACEHOLDERS
            ):
                raise ValueError(
                    "Production ASR API key must be at least 32 characters and not a known placeholder"
                )
            self.api_key = normalized_api_key
            if self.asr_require_model_manifest and not self.asr_model_manifest_path:
                raise ValueError(
                    "Production ASR model manifest path is required when manifest verification is enabled"
                )
        frame_audio_seconds = self.asr_max_frame_bytes / (2 * 16000)
        buffered_audio_seconds = self.asr_ws_max_queue * frame_audio_seconds
        if buffered_audio_seconds > self.asr_max_connection_lag_seconds:
            raise ValueError(
                "WebSocket buffered audio exceeds asr_max_connection_lag_seconds"
            )
        if self.asr_gateway_max_queued_audio_seconds < self.asr_gateway_max_session_buffer_seconds:
            raise ValueError(
                "asr_gateway_max_queued_audio_seconds must cover one session buffer"
            )
        if self.asr_backend == "qwen" and self.asr_stream_mode == "stateful":
            raise ValueError("asr_backend=qwen does not support stateful streaming")
        if self.asr_backend == "faster_whisper" and self.asr_stream_mode != "rolling":
            raise ValueError("asr_backend=faster_whisper requires rolling streaming")
        if self.asr_vad_onset_threshold <= self.asr_vad_offset_threshold:
            raise ValueError(
                "asr_vad_onset_threshold must exceed asr_vad_offset_threshold"
            )
        if self.asr_vad_hangover_ms > self.asr_vad_min_silence_ms:
            raise ValueError(
                "asr_vad_hangover_ms must not exceed asr_vad_min_silence_ms"
            )
        if self.asr_stream_mode == "stateful":
            if self.asr_stream_rollover_seconds <= self.asr_stream_chunk_seconds:
                raise ValueError(
                    "asr_stream_rollover_seconds must exceed the model chunk duration"
                )
            if self.asr_stream_rollover_seconds <= frame_audio_seconds:
                raise ValueError(
                    "asr_stream_rollover_seconds must exceed one transport frame"
                )
            if self.asr_max_utterance_seconds <= self.asr_stream_chunk_seconds:
                raise ValueError(
                    "normal utterance limit must exceed the model chunk duration"
                )
            if self.asr_max_utterance_seconds <= frame_audio_seconds:
                raise ValueError(
                    "normal utterance limit must exceed one transport frame"
                )
            if (
                self.asr_backend == "qwen_vllm"
                and self.asr_max_frame_bytes < self.asr_vad_frame_samples * 2
            ):
                raise ValueError(
                    "asr_max_frame_bytes must hold at least one VAD frame"
                )
            if (
                self.asr_state_watchdog_seconds
                <= self.asr_max_utterance_seconds + frame_audio_seconds
            ):
                raise ValueError(
                    "asr_state_watchdog_seconds must exceed the normal utterance limit plus one frame"
                )
            if self.asr_backend == "qwen_vllm" and self.asr_model_id.rstrip("/").endswith("-hf"):
                raise ValueError("qwen_vllm stateful model_id must not use the -hf export")
            if self.asr_backend == "qwen_vllm" and self.asr_model_id != "Qwen/Qwen3-ASR-1.7B":
                if not self.asr_require_model_manifest or not self.asr_model_manifest_path:
                    raise ValueError(
                        "qwen_vllm requires Qwen/Qwen3-ASR-1.7B or an approved local model with its manifest"
                    )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
