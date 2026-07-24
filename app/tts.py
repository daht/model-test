from __future__ import annotations

import io
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator

from app.config import Settings


class TTSSynthesizer:
    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        raise NotImplementedError

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        raise NotImplementedError


@dataclass
class MockTTSSynthesizer(TTSSynthesizer):
    sample_rate: int

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)),
            sample_rate=self.sample_rate,
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        duration_seconds = min(0.8, max(0.15, len(text) * 0.025))
        sample_count = int(self.sample_rate * duration_seconds)
        frequency = 440.0 if (voice or "default") == "default" else 554.37
        amplitude = 4000
        chunk_samples = max(1, self.sample_rate // 25)

        for chunk_start in range(0, sample_count, chunk_samples):
            pcm = bytearray()
            chunk_end = min(chunk_start + chunk_samples, sample_count)
            for index in range(chunk_start, chunk_end):
                phase = 2.0 * math.pi * frequency * index / self.sample_rate
                sample = int(amplitude * math.sin(phase))
                pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
            yield bytes(pcm)


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
            raise RuntimeError(
                f"CosyVoice backend is unavailable: {self._load_error}"
            ) from self._load_error

        try:
            if self.settings.tts_cosyvoice_repo:
                repo_path = str(Path(self.settings.tts_cosyvoice_repo).expanduser())
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)

            from cosyvoice.cli.cosyvoice import AutoModel

            self._model = AutoModel(model_dir=self.settings.tts_model_id)
            if not hasattr(self._model, "add_zero_shot_spk"):
                raise RuntimeError("CosyVoice model does not support zero-shot speakers")
            added = self._model.add_zero_shot_spk(
                self.settings.tts_prompt_text,
                self.settings.tts_prompt_wav,
                self.settings.tts_default_voice,
            )
            if added is not True:
                raise RuntimeError("failed to register the default zero-shot speaker")
        except Exception as exc:
            self._load_error = exc
            raise RuntimeError(f"CosyVoice backend is unavailable: {exc}") from exc

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        pcm = b"".join(self.stream_pcm(text, voice))
        return _wav_bytes(pcm, sample_rate=self.settings.tts_sample_rate)

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        with self._lock:
            self._load()
            assert self._model is not None

            inference = self._select_inference(
                text,
                voice or self.settings.tts_default_voice,
                stream=True,
            )
            yielded = False
            for chunk in inference:
                pcm = _cosyvoice_chunk_to_pcm(chunk)
                if not pcm:
                    continue
                yielded = True
                yield pcm
            if not yielded:
                raise RuntimeError("CosyVoice did not return audio")

    def _select_inference(self, text: str, voice: str, stream: bool = True):
        if voice != self.settings.tts_default_voice:
            raise RuntimeError(f"unknown TTS voice: {voice}")
        if not hasattr(self._model, "inference_zero_shot"):
            raise RuntimeError("CosyVoice model does not expose zero-shot inference")
        return self._model.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=voice,
            stream=stream,
        )


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

    pcm_chunks = []
    for chunk in result:
        pcm = _cosyvoice_chunk_to_pcm(chunk)
        if pcm:
            pcm_chunks.append(pcm)

    if not pcm_chunks:
        raise RuntimeError("CosyVoice did not return audio")
    return _wav_bytes(b"".join(pcm_chunks), sample_rate=sample_rate)


def _cosyvoice_chunk_to_pcm(chunk) -> bytes:
    audio = chunk.get("tts_speech") if isinstance(chunk, dict) else chunk
    if audio is None:
        raise RuntimeError("CosyVoice returned an audio chunk without tts_speech")
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    if isinstance(audio, bytes):
        if audio.startswith(b"RIFF") and b"WAVE" in audio[:16]:
            with wave.open(io.BytesIO(audio), "rb") as wav_file:
                if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                    raise RuntimeError("CosyVoice WAV chunks must be mono pcm_s16le")
                return wav_file.readframes(wav_file.getnframes())
        return audio
    if hasattr(audio, "dtype"):
        import numpy as np

        samples = np.asarray(audio).reshape(-1)
        if samples.dtype.kind == "f":
            samples = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        else:
            samples = samples.astype("<i2")
        return samples.tobytes()
    raise RuntimeError(
        f"CosyVoice returned unsupported audio type: {type(audio).__name__}"
    )
