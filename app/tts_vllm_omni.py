from __future__ import annotations

from typing import Iterator

import httpx

from app.config import Settings


class VLLMOmniTTSSynthesizer:
    """Streaming client for the vLLM-Omni Qwen3-TTS speech API."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._base_url = settings.tts_vllm_omni_base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=settings.tts_vllm_omni_timeout_seconds
        )

    def start(self) -> None:
        try:
            response = self._client.get(
                f"{self._base_url}/health",
                timeout=self.settings.tts_vllm_omni_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM-Omni TTS backend is unavailable: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        from app.tts import _wav_bytes

        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)),
            sample_rate=self.settings.tts_sample_rate,
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        selected_voice = voice or self.settings.tts_default_voice
        if selected_voice != self.settings.tts_default_voice:
            raise RuntimeError(f"unknown TTS voice: {selected_voice}")

        headers = {}
        if self.settings.tts_vllm_omni_api_key:
            headers["Authorization"] = (
                f"Bearer {self.settings.tts_vllm_omni_api_key}"
            )
        payload = {
            "model": self.settings.tts_vllm_omni_model,
            "input": text,
            "voice": self.settings.tts_qwen_speaker,
            "response_format": "pcm",
            "stream": True,
            "task_type": "CustomVoice",
            "language": self.settings.tts_qwen_language,
        }
        if self.settings.tts_qwen_instruct:
            payload["instructions"] = self.settings.tts_qwen_instruct

        pending = b""
        yielded = False
        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/v1/audio/speech",
                json=payload,
                headers=headers,
                timeout=self.settings.tts_vllm_omni_timeout_seconds,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    chunk = pending + chunk
                    pending = chunk[-1:] if len(chunk) % 2 else b""
                    chunk = chunk[:-1] if pending else chunk
                    if not chunk:
                        continue
                    yielded = True
                    yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM-Omni TTS inference failed: {exc}") from exc

        if pending:
            raise RuntimeError("vLLM-Omni returned an odd-length pcm_s16le stream")
        if not yielded:
            raise RuntimeError("vLLM-Omni returned no audio")
