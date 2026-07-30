"""Streaming client for VoxServe's Qwen3-TTS endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import httpx

from app.config import Settings


class VoxServeTTSSynthesizer:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._base_url = settings.tts_voxserve_base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=settings.tts_voxserve_timeout_seconds)

    def start(self) -> None:
        try:
            response = self._client.get(
                f"{self._base_url}/health", timeout=self.settings.tts_voxserve_timeout_seconds
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"VoxServe TTS backend is unavailable: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def capacity_snapshot(self) -> dict[str, object]:
        return {
            "ready": None,
            "supports_realtime_streaming": True,
            "supports_microbatch": True,
            "queue_depth": None,
            "queue_size": None,
            "batch_size": None,
            "batch_wait_ms": None,
            "dispatched_batches": None,
            "dispatched_requests": None,
            "last_batch_size": None,
        }

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        from app.tts import _wav_bytes

        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)), sample_rate=self.settings.tts_sample_rate
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        selected_voice = voice or self.settings.tts_default_voice
        if selected_voice != self.settings.tts_default_voice:
            raise RuntimeError(f"unknown TTS voice: {selected_voice}")
        if self.settings.tts_voxserve_mode == "base":
            yield from self._stream_base(text)
            return
        yield from self._stream_custom_voice(text)

    def _stream_custom_voice(self, text: str) -> Iterator[bytes]:
        if self.settings.tts_qwen_instruct:
            raise RuntimeError(
                "VoxServe /v1/audio/speech does not expose Qwen instruction control"
            )
        payload = {
            "model": self.settings.tts_voxserve_model,
            "input": text,
            "voice": self.settings.tts_qwen_speaker,
            "language": self.settings.tts_qwen_language,
            "response_format": "pcm",
            "stream": True,
        }
        yield from self._stream_pcm_response(
            f"{self._base_url}/v1/audio/speech", json=payload
        )

    def _stream_base(self, text: str) -> Iterator[bytes]:
        reference_path = Path(self.settings.tts_qwen_reference_wav or "")
        if not reference_path.is_file():
            raise RuntimeError(f"VoxServe Base reference WAV is unavailable: {reference_path}")
        reference_text = (self.settings.tts_qwen_reference_text or "").strip()
        if not reference_text:
            raise RuntimeError("VoxServe Base reference text is empty")

        with reference_path.open("rb") as reference_audio:
            data = {
                "text": text,
                "streaming": "true",
                "language": self.settings.tts_qwen_language,
                "ref_text": reference_text,
            }
            files = {"audio": (reference_path.name, reference_audio, "audio/wav")}
            yield from self._stream_wav_response(
                f"{self._base_url}/generate", data=data, files=files
            )

    def _stream_pcm_response(self, url: str, **request_kwargs) -> Iterator[bytes]:
        pending = b""
        yielded = False
        try:
            with self._client.stream(
                "POST", url, timeout=self.settings.tts_voxserve_timeout_seconds, **request_kwargs
            ) as response:
                response.raise_for_status()
                _require_content_type(response, "audio/pcm", "VoxServe")
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    chunk = pending + chunk
                    pending = chunk[-1:] if len(chunk) % 2 else b""
                    chunk = chunk[:-1] if pending else chunk
                    if chunk:
                        yielded = True
                        yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"VoxServe TTS inference failed: {exc}") from exc
        if pending:
            raise RuntimeError("VoxServe returned an odd-length pcm_s16le stream")
        if not yielded:
            raise RuntimeError("VoxServe returned no audio")

    def _stream_wav_response(self, url: str, **request_kwargs) -> Iterator[bytes]:
        header = b""
        header_complete = False
        pending = b""
        yielded = False
        try:
            with self._client.stream(
                "POST", url, timeout=self.settings.tts_voxserve_timeout_seconds, **request_kwargs
            ) as response:
                response.raise_for_status()
                _require_content_type(response, "audio/wav", "VoxServe")
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if not header_complete:
                        header += chunk
                        header_length = _streaming_wav_header_length(header, self.settings.tts_sample_rate)
                        if header_length is None:
                            continue
                        chunk = header[header_length:]
                        header_complete = True
                    chunk = pending + chunk
                    pending = chunk[-1:] if len(chunk) % 2 else b""
                    chunk = chunk[:-1] if pending else chunk
                    if chunk:
                        yielded = True
                        yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"VoxServe TTS inference failed: {exc}") from exc
        if not header_complete:
            raise RuntimeError("VoxServe returned a truncated WAV stream")
        if pending:
            raise RuntimeError("VoxServe returned an odd-length pcm_s16le stream")
        if not yielded:
            raise RuntimeError("VoxServe returned no audio")


def _require_content_type(response: httpx.Response, expected: str, backend: str) -> None:
    actual = response.headers.get("content-type", "").lower()
    if not actual.startswith(expected):
        raise RuntimeError(
            f"{backend} returned an unexpected content type: expected {expected}, got {actual or 'missing'}"
        )


def _streaming_wav_header_length(data: bytes, expected_sample_rate: int) -> int | None:
    if len(data) < 12:
        return None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError("VoxServe returned an invalid WAV stream")
    offset = 12
    fmt: tuple[int, int, int, int] | None = None
    while True:
        if len(data) < offset + 8:
            return None
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        payload_start = offset + 8
        if chunk_id == b"data":
            if fmt is None:
                raise RuntimeError("VoxServe WAV stream is missing the fmt chunk")
            audio_format, channels, sample_rate, sample_width = fmt
            if (audio_format, channels, sample_rate, sample_width) != (1, 1, expected_sample_rate, 2):
                raise RuntimeError("VoxServe WAV stream must be mono pcm_s16le at the configured sample rate")
            return payload_start
        if len(data) < payload_start + chunk_size:
            return None
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise RuntimeError("VoxServe WAV fmt chunk is invalid")
            audio_format = int.from_bytes(data[payload_start : payload_start + 2], "little")
            channels = int.from_bytes(data[payload_start + 2 : payload_start + 4], "little")
            sample_rate = int.from_bytes(data[payload_start + 4 : payload_start + 8], "little")
            bits_per_sample = int.from_bytes(data[payload_start + 14 : payload_start + 16], "little")
            fmt = (audio_format, channels, sample_rate, bits_per_sample // 8)
        offset = payload_start + chunk_size + (chunk_size % 2)
