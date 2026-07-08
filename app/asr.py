from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.config import Settings


@dataclass
class TranscriptionResult:
    text: str
    language: str | None


class ASRTranscriber:
    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        raise NotImplementedError


@dataclass
class MockASRTranscriber(ASRTranscriber):
    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        detected_language = language or "auto"
        filename = Path(audio_path).name
        if filename.endswith(".stream.wav"):
            filename = "stream.wav"
        elif "-" in filename:
            stem, suffix = filename.rsplit(".", maxsplit=1)
            original_stem = stem.rsplit("-", maxsplit=1)[0]
            filename = f"{original_stem}.{suffix}"
        return TranscriptionResult(
            text=f"[mock asr {detected_language}] {filename}",
            language=detected_language,
        )


class QwenASRTranscriber(ASRTranscriber):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map[self.settings.asr_torch_dtype]

        self._processor = AutoProcessor.from_pretrained(
            self.settings.asr_model_id,
            trust_remote_code=self.settings.trust_remote_code,
        )
        model_kwargs = {
            "device_map": self.settings.asr_device,
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        try:
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.settings.asr_model_id,
                **model_kwargs,
            )
        except TypeError:
            if dtype is not None:
                model_kwargs.pop("dtype", None)
                model_kwargs["torch_dtype"] = dtype
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.settings.asr_model_id,
                **model_kwargs,
            )
        self._model.eval()

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        with self._lock:
            self._load()
            assert self._processor is not None
            assert self._model is not None

            import torch

            request_kwargs = {"audio": audio_path}
            if language:
                request_kwargs["language"] = language

            inputs = self._processor.apply_transcription_request(**request_kwargs)
            inputs = inputs.to(self._model.device, self._model.dtype)

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.asr_max_new_tokens,
                    do_sample=False,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            parsed = self._processor.decode(generated_ids, return_format="parsed")[0]

            if isinstance(parsed, dict):
                return TranscriptionResult(
                    text=parsed.get("transcription", ""),
                    language=parsed.get("language") or language,
                )

            text = self._processor.decode(generated_ids, return_format="transcription_only")[0]
            return TranscriptionResult(text=text, language=language)


def create_asr_transcriber(settings: Settings) -> ASRTranscriber:
    if settings.asr_backend == "mock":
        return MockASRTranscriber()
    return QwenASRTranscriber(settings)
