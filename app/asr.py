from __future__ import annotations

import importlib.metadata
import logging
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.config import Settings
from app.asr_vad import create_vad_endpoint_detector

QWEN_ASR_RUNTIME_VERSION = "0.0.6"
VLLM_RUNTIME_VERSION = "0.14.0"

logger = logging.getLogger(__name__)

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


@dataclass(init=False)
class StreamingTranscriptionResult:
    segment_id: int
    segment_text: str
    language: str | None
    decoded_samples_delta: int | None
    model_updated: bool
    segment_finished: bool
    queue_wait_seconds: float
    inference_seconds: float

    def __init__(
        self,
        text: str = "",
        language: str | None = None,
        processed_samples: int | None = None,
        model_updated: bool = True,
        segment_finished: bool = False,
        *,
        segment_id: int = 0,
        segment_text: str | None = None,
        decoded_samples_delta: int | None = None,
        queue_wait_seconds: float = 0.0,
        inference_seconds: float = 0.0,
    ) -> None:
        self.segment_id = segment_id
        self.segment_text = text if segment_text is None else segment_text
        self.language = language
        self.decoded_samples_delta = (
            processed_samples
            if decoded_samples_delta is None
            else decoded_samples_delta
        )
        self.model_updated = model_updated
        self.segment_finished = segment_finished
        self.queue_wait_seconds = queue_wait_seconds
        self.inference_seconds = inference_seconds

    @property
    def text(self) -> str:
        return self.segment_text

    @property
    def processed_samples(self) -> int | None:
        return self.decoded_samples_delta


class ASRStreamingSession:
    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def finish(self) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def finish_segment(self) -> StreamingTranscriptionResult:
        raise NotImplementedError

    def reset_segment(self) -> StreamingTranscriptionResult:
        return self.finish_segment()

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

    def create_streaming_session(self, language: str | None = None) -> ASRStreamingSession:
        return MockASRStreamingSession(language)


class MockASRStreamingSession(ASRStreamingSession):
    def __init__(self, language: str | None) -> None:
        self.language = language or "auto"
        self._closed = False
        self._segment_id = 0

    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        self._require_open()
        if sample_rate != 16000:
            raise ValueError("mock streaming requires 16000 Hz sample_rate")
        return StreamingTranscriptionResult(
            segment_text="",
            segment_id=self._segment_id,
            language=self.language,
            decoded_samples_delta=len(pcm_bytes) // 2,
            model_updated=True,
        )

    def finish_segment(self) -> StreamingTranscriptionResult:
        self._require_open()
        result = StreamingTranscriptionResult(
            segment_text="",
            segment_id=self._segment_id,
            language=self.language,
            decoded_samples_delta=0,
            model_updated=False,
            segment_finished=True,
        )
        self._segment_id += 1
        return result

    def finish(self) -> StreamingTranscriptionResult:
        self._require_open()
        self._closed = True
        return StreamingTranscriptionResult(
            segment_text="",
            segment_id=self._segment_id,
            language=self.language,
            decoded_samples_delta=0,
            model_updated=False,
        )

    def abort(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming session is closed")


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
        self._closed = False
        self._segment_id = 0
        self._segment_received_samples = 0
        self._undecoded_samples = 0
        self._max_utterance_samples = int(
            round(transcriber.settings.asr_max_utterance_seconds * 16000)
        )
        self._watchdog_samples = int(
            round(transcriber.settings.asr_state_watchdog_seconds * 16000)
        )
        self.state = transcriber._init_streaming_state(language=self.language)

    def add_pcm_s16le(self, pcm_bytes: bytes, sample_rate: int) -> StreamingTranscriptionResult:
        self._require_open()
        if sample_rate != 16000:
            raise ValueError("Qwen vLLM streaming requires 16000 Hz sample_rate")
        if len(pcm_bytes) > self.transcriber.settings.asr_max_frame_bytes:
            raise ValueError("Qwen vLLM streaming frame exceeds configured limit")
        if not pcm_bytes:
            return StreamingTranscriptionResult(
                segment_id=self._segment_id,
                segment_text=self._segment_text(),
                language=getattr(self.state, "language", self.language),
                decoded_samples_delta=0,
                model_updated=False,
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
        if self._segment_received_samples + sample_count > self._watchdog_samples:
            raise RuntimeError("ASR utterance state exceeded invariant watchdog")
        previous_chunk_id = self._chunk_id()
        previous_text = self._segment_text()
        self._segment_received_samples += sample_count
        self._undecoded_samples += sample_count
        state = self.transcriber._streaming_transcribe(pcm, self.state)
        self.state = state
        decoded_samples, model_updated = self._decode_progress(
            previous_chunk_id, previous_text
        )
        if self._segment_received_samples >= self._max_utterance_samples:
            flush_samples, flush_updated = self._finish_active_state()
            decoded_samples += flush_samples
            model_updated = model_updated or flush_updated
            text = self._segment_text()
            language = getattr(self.state, "language", self.language)
            result = StreamingTranscriptionResult(
                segment_id=self._segment_id,
                segment_text=text,
                language=language,
                decoded_samples_delta=decoded_samples,
                model_updated=model_updated,
                segment_finished=True,
            )
            self._start_fresh_segment()
            return result
        return StreamingTranscriptionResult(
            segment_id=self._segment_id,
            segment_text=self._segment_text(),
            language=getattr(state, "language", self.language),
            decoded_samples_delta=decoded_samples,
            model_updated=model_updated,
        )

    def finish(self) -> StreamingTranscriptionResult:
        self._require_open()
        processed_samples, model_updated = self._finish_active_state()
        self._closed = True
        return StreamingTranscriptionResult(
            segment_id=self._segment_id,
            segment_text=self._segment_text(),
            language=getattr(self.state, "language", self.language),
            decoded_samples_delta=processed_samples,
            model_updated=model_updated,
        )

    def finish_segment(self) -> StreamingTranscriptionResult:
        self._require_open()
        processed_samples, model_updated = self._finish_active_state()
        text = self._segment_text()
        language = getattr(self.state, "language", self.language)
        result = StreamingTranscriptionResult(
            segment_id=self._segment_id,
            segment_text=text,
            language=language,
            decoded_samples_delta=processed_samples,
            model_updated=model_updated,
            segment_finished=True,
        )
        self._start_fresh_segment()
        return result

    def reset_segment(self) -> StreamingTranscriptionResult:
        return self.finish_segment()

    def abort(self) -> None:
        self.state = None
        self._closed = True

    def _segment_text(self) -> str:
        return getattr(self.state, "text", "")

    def _chunk_id(self) -> int:
        return int(getattr(self.state, "chunk_id", 0))

    def _decode_progress(
        self, previous_chunk_id: int, previous_text: str
    ) -> tuple[int, bool]:
        decoded_chunks = max(0, self._chunk_id() - previous_chunk_id)
        chunk_samples = int(
            getattr(
                self.state,
                "chunk_size_samples",
                round(self.transcriber.settings.asr_stream_chunk_seconds * 16000),
            )
        )
        decoded_samples = min(self._undecoded_samples, decoded_chunks * chunk_samples)
        self._undecoded_samples -= decoded_samples
        return decoded_samples, decoded_chunks > 0 or self._segment_text() != previous_text

    def _finish_active_state(self) -> tuple[int, bool]:
        previous_chunk_id = self._chunk_id()
        previous_text = self._segment_text()
        self.state = self.transcriber._finish_streaming_transcribe(self.state)
        model_updated = (
            self._chunk_id() > previous_chunk_id
            or self._segment_text() != previous_text
        )
        processed_samples = self._undecoded_samples if model_updated else 0
        self._undecoded_samples -= processed_samples
        return processed_samples, model_updated

    def _start_fresh_segment(self) -> None:
        self._segment_id += 1
        self.state = self.transcriber._init_streaming_state(language=self.language)
        self._segment_received_samples = 0
        self._undecoded_samples = 0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming session is closed")


class QwenVLLMASRTranscriber(ASRTranscriber):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._model = None
        self._streaming_warmup_complete = False

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
        if self.settings.asr_stream_mode != "stateful":
            return
        if self._streaming_warmup_complete:
            return
        self._validate_qwen_runtime_contract()
        create_vad_endpoint_detector(self.settings)
        warmup_samples = max(
            16000,
            int(math.ceil(self.settings.asr_stream_chunk_seconds * 16000)),
        )
        pcm = array(
            "h",
            (
                int(8000 * math.sin(2 * math.pi * 440 * index / 16000))
                for index in range(warmup_samples)
            ),
        ).tobytes()
        logger.info(
            "asr_streaming_warmup_started backend=qwen_vllm audio_samples=%d",
            warmup_samples,
        )
        session = self.create_streaming_session(language="English")
        try:
            decoded_samples = 0
            model_updated = False
            frame_bytes = self.settings.asr_max_frame_bytes
            for offset in range(0, len(pcm), frame_bytes):
                update = session.add_pcm_s16le(
                    pcm[offset : offset + frame_bytes], 16000
                )
                decoded_samples += update.decoded_samples_delta or 0
                model_updated = model_updated or update.model_updated
            final = session.finish()
            decoded_samples += final.decoded_samples_delta or 0
            model_updated = model_updated or final.model_updated
            if decoded_samples <= 0 or not model_updated:
                raise RuntimeError(
                    "Qwen streaming warmup produced no decoded audio progress"
                )
        except Exception:
            session.abort()
            raise
        self._streaming_warmup_complete = True
        logger.info(
            "asr_streaming_warmup_completed backend=qwen_vllm decoded_samples=%d",
            decoded_samples,
        )

    @property
    def streaming_warmup_complete(self) -> bool:
        return self._streaming_warmup_complete

    def _validate_qwen_runtime_contract(self) -> None:
        versions = {
            "qwen-asr": QWEN_ASR_RUNTIME_VERSION,
            "vllm": VLLM_RUNTIME_VERSION,
        }
        for distribution, expected in versions.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"required ASR runtime distribution is missing: {distribution}"
                ) from exc
            if actual != expected:
                raise RuntimeError(
                    f"unsupported {distribution} version: expected {expected}, got {actual}"
                )

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
