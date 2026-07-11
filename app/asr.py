from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.config import Settings

QWEN_VLLM_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cmn": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "zh-yue": "Cantonese",
    "cantonese": "Cantonese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "id": "Indonesian",
    "in": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "kr": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "jp": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "tl": "Filipino",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


@dataclass
class TranscriptionResult:
    text: str
    language: str | None


@dataclass
class StreamingTranscriptionResult:
    text: str
    language: str | None


class ASRStreamingSession:
    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def finish(self) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def reset_segment(self) -> None:
        raise NotImplementedError

    def abort(self) -> None:
        raise NotImplementedError


class ASRTranscriber:
    def warmup(self) -> None:
        raise NotImplementedError

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        raise NotImplementedError

    def create_streaming_session(self, language: str | None = None) -> ASRStreamingSession:
        raise NotImplementedError("This ASR backend does not support stateful streaming")


@dataclass
class MockASRTranscriber(ASRTranscriber):
    def warmup(self) -> None:
        return None

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

    def warmup(self) -> None:
        with self._lock:
            self._load()

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


class QwenVLLMStreamingSession(ASRStreamingSession):
    def __init__(self, transcriber: "QwenVLLMASRTranscriber", language: str | None = None) -> None:
        self.transcriber = transcriber
        self.language = transcriber.normalize_language(language)
        self._text_prefix = ""
        self._closed = False
        self.state = transcriber._init_streaming_state(language=self.language)

    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        self._require_open()
        if sample_rate != 16000:
            raise ValueError("Qwen vLLM streaming requires 16000 Hz sample_rate")
        if not pcm_bytes:
            return StreamingTranscriptionResult(
                text=self._combined_text(),
                language=getattr(self.state, "language", self.language),
            )

        sample_count = len(pcm_bytes) // 2
        try:
            import numpy as np

            pcm = np.frombuffer(pcm_bytes[: sample_count * 2], dtype="<i2")
        except ModuleNotFoundError:
            import sys
            from array import array

            pcm = array("h")
            pcm.frombytes(pcm_bytes[: sample_count * 2])
            if sys.byteorder != "little":
                pcm.byteswap()
        state = self.transcriber._streaming_transcribe(pcm, self.state)
        self.state = state
        return StreamingTranscriptionResult(
            text=self._combined_text(),
            language=getattr(state, "language", self.language),
        )

    def finish(self) -> StreamingTranscriptionResult:
        self._require_open()
        state = self.transcriber._finish_streaming_transcribe(self.state)
        self.state = state
        self._closed = True
        return StreamingTranscriptionResult(
            text=self._combined_text(),
            language=getattr(state, "language", self.language),
        )

    def reset_segment(self) -> None:
        self._require_open()
        self._text_prefix = self._combined_text()
        self.state = self.transcriber._init_streaming_state(language=self.language)

    def abort(self) -> None:
        self.state = None
        self._closed = True

    def _combined_text(self) -> str:
        return self._text_prefix + getattr(self.state, "text", "")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming session is closed")


class QwenVLLMASRTranscriber(ASRTranscriber):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.LLM(
            model=self.settings.asr_model_id,
            gpu_memory_utilization=self.settings.asr_vllm_gpu_memory_utilization,
            max_new_tokens=self.settings.asr_vllm_max_new_tokens,
        )

    def warmup(self) -> None:
        with self._lock:
            self._load()

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        normalized_language = self.normalize_language(language)
        with self._lock:
            self._load()
            assert self._model is not None
            kwargs = {"audio": audio_path}
            if normalized_language:
                kwargs["language"] = normalized_language
            results = self._model.transcribe(**kwargs)
            result = results[0] if isinstance(results, list) else results
            return TranscriptionResult(
                text=getattr(result, "text", ""),
                language=getattr(result, "language", normalized_language),
            )

    def create_streaming_session(self, language: str | None = None) -> ASRStreamingSession:
        return QwenVLLMStreamingSession(self, language=language)

    def normalize_language(self, language: str | None) -> str | None:
        if language is None:
            return None

        stripped = str(language).strip()
        if not stripped:
            return None

        key = stripped.lower().replace("_", "-")
        if key in QWEN_VLLM_LANGUAGE_ALIASES:
            return QWEN_VLLM_LANGUAGE_ALIASES[key]

        primary_subtag = key.split("-", maxsplit=1)[0]
        if primary_subtag in QWEN_VLLM_LANGUAGE_ALIASES:
            return QWEN_VLLM_LANGUAGE_ALIASES[primary_subtag]

        return stripped[:1].upper() + stripped[1:].lower()

    def _init_streaming_state(self, language: str | None = None):
        normalized_language = self.normalize_language(language)
        with self._lock:
            self._load()
            assert self._model is not None
            return self._model.init_streaming_state(
                language=normalized_language,
                unfixed_chunk_num=self.settings.asr_stream_unfixed_chunk_num,
                unfixed_token_num=self.settings.asr_stream_unfixed_token_num,
                chunk_size_sec=self.settings.asr_stream_chunk_seconds,
            )

    def _streaming_transcribe(self, pcm, state):
        with self._lock:
            assert self._model is not None
            return self._model.streaming_transcribe(pcm, state)

    def _finish_streaming_transcribe(self, state):
        with self._lock:
            assert self._model is not None
            return self._model.finish_streaming_transcribe(state)


def create_asr_transcriber(settings: Settings) -> ASRTranscriber:
    if settings.asr_backend == "mock":
        return MockASRTranscriber()
    if settings.asr_backend == "qwen_vllm":
        return QwenVLLMASRTranscriber(settings)
    return QwenASRTranscriber(settings)
