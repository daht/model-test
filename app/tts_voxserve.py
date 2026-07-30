"""Streaming client for VoxServe's Qwen3-TTS endpoints."""

from __future__ import annotations

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

def _require_content_type(response: httpx.Response, expected: str, backend: str) -> None:
    actual = response.headers.get("content-type", "").lower()
    if not actual.startswith(expected):
        raise RuntimeError(
            f"{backend} returned an unexpected content type: expected {expected}, got {actual or 'missing'}"
        )
