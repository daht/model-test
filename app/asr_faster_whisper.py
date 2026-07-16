from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from app.asr_artifacts import verify_model_manifest
from app.asr_gateway_backends import (
    AdapterResult,
    BackendCapabilities,
    DispatchMode,
    ResultMode,
    StreamingMode,
    VadMode,
)
from app.asr_gateway_scheduler import InferenceJob, InferenceResult


@dataclass(frozen=True)
class DecodedText:
    text: str
    language: str


class FasterWhisperEngine:
    """Pinned faster-whisper 1.2.1 batched generation boundary."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        compute_type: str = "float16",
    ) -> None:
        from faster_whisper import WhisperModel
        from faster_whisper.transcribe import BatchedInferencePipeline

        runtime_device, device_index = _runtime_device(device)
        self._model = WhisperModel(
            model_id,
            device=runtime_device,
            device_index=device_index,
            compute_type=compute_type,
            num_workers=1,
        )
        self._pipeline = BatchedInferencePipeline(self._model)

    def warmup(self) -> None:
        samples = np.arange(8_000, dtype=np.float32)
        waveform = (0.01 * np.sin(2 * np.pi * 440 * samples / 16_000)).astype(
            np.float32
        )
        self.transcribe_batch([waveform], language="en", beam_size=1)

    def transcribe_batch(
        self,
        audio: Sequence[np.ndarray],
        *,
        language: str | None,
        beam_size: int,
    ) -> list[DecodedText]:
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import (
            TranscriptionOptions,
            get_suppressed_tokens,
        )

        if not audio:
            return []
        features = np.stack(
            [
                pad_or_trim(self._model.feature_extractor(item)[..., :-1])
                for item in audio
            ]
        )
        tokenizer = Tokenizer(
            self._model.hf_tokenizer,
            self._model.model.is_multilingual,
            task="transcribe",
            language=language or "en",
        )
        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=1,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
            without_timestamps=True,
            max_initial_timestamp=0.0,
            word_timestamps=False,
            prepend_punctuations='"\'“¿([{-',
            append_punctuations='"\'.。,，!！?？:：”)]}、',
            multilingual=language is None,
            max_new_tokens=None,
            clip_timestamps=[],
            hallucination_silence_threshold=None,
            hotwords=None,
        )
        encoder_output, outputs = self._pipeline.generate_segment_batched(
            features, tokenizer, options
        )
        if language is None:
            detected = [
                candidates[0][0][2:-2]
                for candidates in self._model.model.detect_language(encoder_output)
            ]
        else:
            detected = [language] * len(outputs)
        return [
            DecodedText(tokenizer.decode(output["tokens"]).strip(), item_language)
            for output, item_language in zip(outputs, detected)
        ]

    def close(self) -> None:
        self._pipeline = None
        self._model = None


@dataclass
class _SessionState:
    backend_session_id: str
    language: str | None
    pcm: bytearray
    cached_final: str | None = None


class FasterWhisperAdapter:
    def __init__(
        self,
        engine_factory: Callable[[], Any],
        *,
        worker_id: str,
        model_id: str,
        model_revision: str,
        gpu_id: str,
        session_capacity: int,
        batch_size: int,
        partial_beam_size: int,
        final_beam_size: int,
        max_segment_samples: int,
        model_manifest_path: str | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._sessions: dict[str, _SessionState] = {}
        self._engine_lock = asyncio.Lock()
        self._ready = False
        self._partial_beam_size = partial_beam_size
        self._final_beam_size = final_beam_size
        self._max_segment_samples = max_segment_samples
        self._model_manifest_path = model_manifest_path
        self.capabilities = BackendCapabilities(
            protocol_version=1,
            worker_id=worker_id,
            model_id=model_id,
            model_revision=model_revision,
            gpu_id=gpu_id,
            languages=("auto",),
            tasks=("transcribe",),
            streaming_mode=StreamingMode.ROLLING,
            dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH,
            vad_mode=VadMode.GATEWAY,
            result_mode=ResultMode.REPLACEABLE_SEGMENT,
            preferred_chunk_samples=24_000,
            max_input_samples=max_segment_samples,
            max_segment_samples=max_segment_samples,
            max_batch_items=batch_size,
            max_batch_samples=max_segment_samples * batch_size,
            max_in_flight=1,
            session_capacity=session_capacity,
            retry_safe=False,
            warmed=False,
            backend_id="local",
        )

    async def warmup(self) -> None:
        if self._model_manifest_path:
            await asyncio.to_thread(
                verify_model_manifest,
                self.capabilities.model_id,
                self._model_manifest_path,
            )
        if self._engine is None:
            self._engine = self._engine_factory()
        await asyncio.to_thread(self._engine.warmup)
        self._ready = True
        self.capabilities = replace(self.capabilities, warmed=True)

    async def open_session(
        self,
        session_id: str,
        *,
        language: str | None = None,
        task: str = "transcribe",
        timestamps: bool = False,
        **_: Any,
    ) -> str:
        self._require_ready()
        if task != "transcribe":
            raise ValueError("faster-whisper streaming only supports transcription")
        if timestamps:
            raise ValueError("faster-whisper streaming timestamps are disabled")
        if session_id in self._sessions:
            raise ValueError("duplicate gateway session")
        backend_session_id = f"fw-{session_id}"
        self._sessions[session_id] = _SessionState(
            backend_session_id,
            None if language in {None, "auto"} else language,
            bytearray(),
        )
        return backend_session_id

    async def submit(
        self, jobs: Sequence[InferenceJob]
    ) -> Sequence[InferenceResult]:
        engine = self._require_ready()
        if not jobs or len(jobs) > self.capabilities.max_batch_items:
            raise ValueError("faster-whisper batch size is outside adapter limits")
        seen_sessions: set[str] = set()
        states: list[_SessionState] = []
        for job in jobs:
            state = self._session(job.session_id)
            if state.backend_session_id != job.backend_session_id:
                raise KeyError("stale session backend identity")
            if job.session_id in seen_sessions:
                raise ValueError("one batch cannot contain two jobs for one session")
            if len(state.pcm) // 2 + job.sample_count > self._max_segment_samples:
                raise BufferError("faster-whisper utterance limit exceeded")
            seen_sessions.add(job.session_id)
            states.append(state)
        for state, job in zip(states, jobs):
            state.pcm.extend(job.pcm)

        grouped: dict[tuple[str | None, int], list[int]] = {}
        for index, (state, job) in enumerate(zip(states, jobs)):
            beam_size = (
                self._final_beam_size if job.final else self._partial_beam_size
            )
            grouped.setdefault((state.language, beam_size), []).append(index)

        decoded: list[DecodedText | None] = [None] * len(jobs)
        async with self._engine_lock:
            for (language, beam_size), indices in grouped.items():
                waveforms = [_pcm_to_float32(states[index].pcm) for index in indices]
                items = await asyncio.to_thread(
                    engine.transcribe_batch,
                    waveforms,
                    language=language,
                    beam_size=beam_size,
                )
                if len(items) != len(indices):
                    raise RuntimeError("faster-whisper result count mismatch")
                for index, item in zip(indices, items):
                    decoded[index] = item

        results = []
        for state, job, item in zip(states, jobs, decoded):
            if item is None:
                raise RuntimeError("faster-whisper omitted a batch result")
            if state.language is None:
                state.language = item.language
            if job.final:
                state.cached_final = item.text
            results.append(
                InferenceResult.from_job(
                    job,
                    text=item.text,
                    tail_text=item.text,
                    segment_id=1,
                    final=job.final,
                )
            )
        return results

    async def finish_segment(self, session_id: str) -> AdapterResult:
        state = self._session(session_id)
        text = await self._consume_final(state)
        return AdapterResult(
            backend_session_id=state.backend_session_id,
            text=text,
            tail_text=text,
        )

    async def finish_session(self, session_id: str) -> AdapterResult:
        state = self._session(session_id)
        text = await self._consume_final(state)
        self._sessions.pop(session_id, None)
        return AdapterResult(
            backend_session_id=state.backend_session_id,
            text=text,
            tail_text=text,
            is_final=True,
        )

    async def abort_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def cancel(self, job_id: str) -> None:
        return None

    async def drain(self) -> None:
        self._ready = False

    async def close(self) -> None:
        self._ready = False
        self._sessions.clear()
        if self._engine is not None:
            await asyncio.to_thread(self._engine.close)
            self._engine = None
        self.capabilities = replace(self.capabilities, warmed=False)

    async def snapshot(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "accepting": self._ready,
            "active_sessions": len(self._sessions),
            "capacity": self.capabilities.session_capacity,
            "session_audio_samples": sum(
                len(state.pcm) // 2 for state in self._sessions.values()
            ),
        }

    async def _consume_final(self, state: _SessionState) -> str:
        if state.cached_final is not None:
            text = state.cached_final
        elif state.pcm:
            engine = self._require_ready()
            async with self._engine_lock:
                items = await asyncio.to_thread(
                    engine.transcribe_batch,
                    [_pcm_to_float32(state.pcm)],
                    language=state.language,
                    beam_size=self._final_beam_size,
                )
            if len(items) != 1:
                raise RuntimeError("faster-whisper final result count mismatch")
            text = items[0].text
            if state.language is None:
                state.language = items[0].language
        else:
            text = ""
        state.pcm.clear()
        state.cached_final = None
        return text

    def _require_ready(self) -> Any:
        if not self._ready or self._engine is None:
            raise RuntimeError("faster-whisper adapter is not ready")
        return self._engine

    def _session(self, session_id: str) -> _SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError("stale session") from exc


def _pcm_to_float32(pcm: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _runtime_device(device: str) -> tuple[str, int | list[int]]:
    normalized = device.strip().lower()
    if normalized.startswith("cuda:"):
        return "cuda", int(normalized.split(":", 1)[1])
    return normalized, 0
