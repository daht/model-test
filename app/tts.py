from __future__ import annotations

import io
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.config import Settings


class TTSSynthesizer:
    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        raise NotImplementedError


@dataclass
class MockTTSSynthesizer(TTSSynthesizer):
    sample_rate: int

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        duration_seconds = min(0.8, max(0.15, len(text) * 0.025))
        sample_count = int(self.sample_rate * duration_seconds)
        frequency = 440.0 if (voice or "default") == "default" else 554.37
        amplitude = 4000

        pcm = bytearray()
        for index in range(sample_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency * index / self.sample_rate))
            pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))

        return _wav_bytes(bytes(pcm), sample_rate=self.sample_rate)


class CosyVoiceTTSSynthesizer(TTSSynthesizer):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._model = None
        self._load_error: Exception | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(f"CosyVoice backend is unavailable: {self._load_error}") from self._load_error

        try:
            if self.settings.tts_cosyvoice_repo:
                repo_path = str(Path(self.settings.tts_cosyvoice_repo).expanduser())
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)

            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoiceModel
            except ImportError:
                from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceModel

            self._model = CosyVoiceModel(self.settings.tts_model_id)
        except Exception as exc:
            self._load_error = exc
            raise RuntimeError(f"CosyVoice backend is unavailable: {exc}") from exc

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        with self._lock:
            self._load()
            assert self._model is not None

            inference = self._select_inference(text, voice or self.settings.tts_default_voice)
            return _cosyvoice_result_to_wav(inference, sample_rate=self.settings.tts_sample_rate)

    def _select_inference(self, text: str, voice: str):
        if not hasattr(self._model, "inference_sft"):
            raise RuntimeError("CosyVoice model does not expose inference_sft")
        return self._model.inference_sft(text, voice, stream=False)


def create_tts_synthesizer(settings: Settings) -> TTSSynthesizer:
    if settings.tts_backend == "mock":
        return MockTTSSynthesizer(sample_rate=settings.tts_sample_rate)
    return CosyVoiceTTSSynthesizer(settings)


def _wav_bytes(pcm_s16le: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_s16le)
    return output.getvalue()


def _cosyvoice_result_to_wav(result, sample_rate: int) -> bytes:
    if isinstance(result, bytes):
        if result.startswith(b"RIFF") and b"WAVE" in result[:16]:
            return result
        return _wav_bytes(result, sample_rate=sample_rate)

    for chunk in result:
        audio = chunk.get("tts_speech") if isinstance(chunk, dict) else chunk
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        if hasattr(audio, "tobytes"):
            return _wav_bytes(audio.tobytes(), sample_rate=sample_rate)
        if isinstance(audio, bytes):
            return _wav_bytes(audio, sample_rate=sample_rate)

    raise RuntimeError("CosyVoice did not return audio")
