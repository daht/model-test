from __future__ import annotations

from typing import Iterator

import numpy as np

from app.config import Settings


class Qwen3TTSSynthesizer:
    """Deployment-scoped Qwen3-TTS CustomVoice synthesizer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        from app.tts import _wav_bytes

        return _wav_bytes(
            b"".join(self.stream_pcm(text, voice)),
            sample_rate=self.settings.tts_sample_rate,
        )

    def stream_pcm(self, text: str, voice: str | None = None) -> Iterator[bytes]:
        if voice not in (None, self.settings.tts_default_voice):
            raise RuntimeError(f"unknown TTS voice: {voice}")

        model = self._load_model()
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=self.settings.tts_qwen_language,
                speaker=self.settings.tts_qwen_speaker,
                instruct=self.settings.tts_qwen_instruct or None,
            )
        except Exception as exc:
            raise RuntimeError(f"Qwen3-TTS inference failed: {exc}") from exc

        if int(sample_rate) != self.settings.tts_sample_rate:
            raise RuntimeError(
                f"Qwen3-TTS returned {sample_rate} Hz; "
                f"configured output is {self.settings.tts_sample_rate} Hz"
            )
        if not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        pcm = _float_audio_to_pcm(wavs[0])
        if pcm:
            # qwen-tts currently exposes complete waveform generation through
            # the Python API; the common stream contract still stays intact.
            yield pcm

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "TTS_BACKEND=qwen requires the qwen-tts package"
            ) from exc

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.settings.torch_dtype, torch.bfloat16)
        try:
            self._model = _load_qwen_model(
                Qwen3TTSModel,
                self.settings.tts_model_id,
                device_map=self.settings.tts_device,
                dtype=dtype,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Qwen3-TTS model {self.settings.tts_model_id}: {exc}"
            ) from exc
        return self._model


def _load_qwen_model(model_class, model_id: str, *, device_map: str, dtype):
    try:
        return model_class.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
        )
    except Exception as exc:
        if "Cannot copy out of meta tensor" not in str(exc):
            raise
        if device_map != "auto":
            try:
                return model_class.from_pretrained(
                    model_id,
                    device_map="auto",
                    dtype=dtype,
                )
            except Exception as auto_exc:
                if "Cannot copy out of meta tensor" not in str(auto_exc):
                    raise
                exc = auto_exc

        try:
            import torch

            loaded = model_class.from_pretrained(
                model_id,
                device_map=None,
                dtype=dtype,
                low_cpu_mem_usage=False,
            )
            target = device_map
            if target == "auto":
                target = "cuda:0" if torch.cuda.is_available() else "cpu"
            if target not in (None, "cpu"):
                model = getattr(loaded, "model", loaded)
                model.to(target)
                if hasattr(loaded, "device"):
                    loaded.device = torch.device(target)
            return loaded
        except Exception as fallback_exc:
            raise RuntimeError(
                "Qwen3-TTS meta-tensor fallback failed after CPU loading"
            ) from fallback_exc


def _float_audio_to_pcm(audio) -> bytes:
    samples = np.asarray(audio).reshape(-1)
    if samples.dtype.kind == "f":
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    else:
        samples = samples.astype("<i2")
    return samples.tobytes()
