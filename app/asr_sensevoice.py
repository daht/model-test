from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
from app.asr_gateway_scheduler import BatchKey, InferenceJob, InferenceResult
from app.asr_observability import CapacityBufferError, events, maximum_character_run, stable_batch_id


@dataclass(frozen=True)
class SenseVoiceDecoded:
    text: str
    metadata: dict[str, str]


LANGUAGE_TAGS = {"zh", "yue", "en", "ja", "ko"}
EMOTION_TAGS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}
EVENT_TAGS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Cough", "Sneeze"}
TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")


def normalize_sensevoice_output(raw: str) -> SenseVoiceDecoded:
    metadata: dict[str, str] = {}
    for tag in TAG_PATTERN.findall(raw):
        if tag in LANGUAGE_TAGS and "language" not in metadata:
            metadata["language"] = tag
        elif tag in EMOTION_TAGS and "emotion" not in metadata:
            metadata["emotion"] = tag.lower()
        elif tag in EVENT_TAGS and "audio_event" not in metadata:
            metadata["audio_event"] = tag.lower()
    return SenseVoiceDecoded(TAG_PATTERN.sub("", raw).strip(), metadata)


class SenseVoiceEngine:
    def __init__(self, model_id: str, *, device: str = "auto", use_itn: bool = True) -> None:
        from funasr import AutoModel

        self._model_id = model_id
        self._use_itn = use_itn
        self._model = AutoModel(
            model=model_id,
            device=device,
            trust_remote_code=False,
            disable_update=True,
        )

    def transcribe_batch(
        self,
        audio: Sequence[np.ndarray],
        *,
        language: str | None,
    ) -> list[SenseVoiceDecoded]:
        if not audio:
            return []
        return self._generate(list(audio), language=language or "auto")

    def warmup(self) -> bytes:
        sample = Path(self._model_id) / "example" / "en.mp3"
        if not sample.is_file():
            raise RuntimeError("SenseVoice warmup speech sample is missing")
        import librosa

        waveform, sample_rate = librosa.load(
            str(sample), sr=16_000, mono=True, dtype=np.float32
        )
        if (
            sample_rate != 16_000
            or waveform.ndim != 1
            or not waveform.size
            or not np.isfinite(waveform).all()
            or float(np.max(np.abs(waveform))) == 0.0
        ):
            raise RuntimeError("SenseVoice warmup speech sample is invalid")
        samples = np.clip(
            np.rint(waveform * 32768.0), -32768, 32767
        ).astype("<i2")
        return samples.tobytes()

    def _generate(self, inputs: list[Any], *, language: str) -> list[SenseVoiceDecoded]:
        raw = self._model.generate(
            input=inputs,
            cache={},
            language=language,
            use_itn=self._use_itn,
            batch_size=len(inputs),
        )
        if not isinstance(raw, list) or len(raw) != len(inputs):
            raise RuntimeError("SenseVoice result contract mismatch")
        decoded: list[SenseVoiceDecoded] = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise RuntimeError("SenseVoice result contract mismatch")
            decoded.append(normalize_sensevoice_output(item["text"]))
        return decoded

    def close(self) -> None:
        self._model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass
class _SessionState:
    backend_session_id: str
    language: str | None
    pcm: bytearray
    cached_final: SenseVoiceDecoded | None = None


class SenseVoiceAdapter:
    def __init__(
        self,
        engine_factory: Any,
        *,
        worker_id: str,
        model_id: str,
        model_revision: str,
        gpu_id: str,
        session_capacity: int,
        batch_size: int,
        max_segment_samples: int,
        model_manifest_path: str | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._sessions: dict[str, _SessionState] = {}
        self._engine_lock = asyncio.Lock()
        self._ready = False
        self._max_segment_samples = max_segment_samples
        self._model_manifest_path = model_manifest_path
        self._engine_observer: Any = None
        self._capacity_observer: Any = None
        self.capabilities = BackendCapabilities(
            protocol_version=1,
            worker_id=worker_id,
            model_id=model_id,
            model_revision=model_revision,
            gpu_id=gpu_id,
            languages=("auto", "zh", "yue", "en", "ja", "ko"),
            tasks=("transcribe",),
            streaming_mode=StreamingMode.ROLLING,
            dispatch_mode=DispatchMode.DYNAMIC_MICROBATCH,
            vad_mode=VadMode.GATEWAY,
            result_mode=ResultMode.REPLACEABLE_SEGMENT,
            preferred_chunk_samples=min(32_000, max_segment_samples),
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

    def set_engine_observer(self, observer: Any) -> None:
        self._engine_observer = observer

    def set_capacity_observer(self, observer: Any) -> None:
        self._capacity_observer = observer

    async def warmup(self) -> None:
        if self._model_manifest_path:
            await asyncio.to_thread(
                verify_model_manifest,
                self.capabilities.model_id,
                self._model_manifest_path,
            )
        if self._engine is None:
            self._engine = self._engine_factory()
        pcm = await asyncio.to_thread(self._engine.warmup)
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise RuntimeError("SenseVoice warmup PCM is invalid")
        pcm = pcm[: self._max_segment_samples * 2]
        session_id = "__sensevoice_warmup__"
        self._ready = True
        try:
            backend_session_id = await self.open_session(session_id, language="en")
            samples = len(pcm) // 2
            job = InferenceJob(
                job_id="sensevoice-warmup",
                session_id=session_id,
                generation=0,
                job_sequence=1,
                worker_id=self.capabilities.worker_id,
                backend_session_id=backend_session_id,
                start_sample=0,
                end_sample=samples,
                pcm=pcm,
                deadline=float("inf"),
                batch_key=BatchKey(
                    self.capabilities.worker_id,
                    self.capabilities.model_revision,
                    "en",
                    "transcribe",
                    False,
                    "",
                    "final:warmup",
                    "pcm_s16le",
                    0,
                ),
                final=True,
            )
            result = (await self.submit([job]))[0]
            await self.finish_session(session_id)
            if (
                not result.text
                or result.metadata is None
                or result.metadata.get("language") != "en"
            ):
                raise RuntimeError("SenseVoice warmup speech result is invalid")
        except Exception:
            self._ready = False
            self._sessions.pop(session_id, None)
            raise
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
            raise ValueError("SenseVoice streaming only supports transcription")
        if timestamps:
            raise ValueError("SenseVoice streaming timestamps are disabled")
        normalized_language = language or "auto"
        if normalized_language not in self.capabilities.languages:
            raise ValueError("SenseVoice streaming language is unsupported")
        if session_id in self._sessions:
            raise ValueError("duplicate gateway session")
        backend_session_id = f"sv-{session_id}"
        self._sessions[session_id] = _SessionState(
            backend_session_id,
            None if normalized_language == "auto" else normalized_language,
            bytearray(),
        )
        return backend_session_id

    async def submit(self, jobs: Sequence[InferenceJob]) -> Sequence[InferenceResult]:
        engine = self._require_ready()
        if not jobs or len(jobs) > self.capabilities.max_batch_items:
            raise ValueError("SenseVoice batch size is outside adapter limits")
        seen_sessions: set[str] = set()
        states: list[_SessionState] = []
        for job in jobs:
            state = self._session(job.session_id)
            if state.backend_session_id != job.backend_session_id:
                raise KeyError("stale session backend identity")
            if job.session_id in seen_sessions:
                raise ValueError("one batch cannot contain two jobs for one session")
            current_samples = len(state.pcm) // 2
            if current_samples + job.sample_count > self._max_segment_samples:
                error = CapacityBufferError(
                    "adapter_utterance_limit",
                    limit=self._max_segment_samples,
                    current=current_samples,
                    incoming=job.sample_count,
                    message="SenseVoice utterance limit exceeded",
                )
                events().emit(
                    "asr_buffer_rejected",
                    component="sensevoice_adapter",
                    session_id=job.session_id,
                    generation=job.generation,
                    job_id=job.job_id,
                    reason=error.reason,
                    **error.safe_fields,
                )
                if self._capacity_observer is not None:
                    self._capacity_observer(error.reason)
                raise error
            seen_sessions.add(job.session_id)
            states.append(state)
        for state, job in zip(states, jobs):
            state.pcm.extend(job.pcm)

        grouped: dict[str | None, list[int]] = {}
        for index, state in enumerate(states):
            grouped.setdefault(state.language, []).append(index)
        decoded: list[SenseVoiceDecoded | None] = [None] * len(jobs)
        batch_id = stable_batch_id(jobs)
        async with self._engine_lock:
            for group_ordinal, (language, indices) in enumerate(grouped.items(), start=1):
                waveforms = [_pcm_to_float32(states[index].pcm) for index in indices]
                accumulated = [len(item) / 16_000 for item in waveforms]
                started = time.monotonic()
                items = await asyncio.to_thread(
                    engine.transcribe_batch,
                    waveforms,
                    language=language,
                )
                elapsed = time.monotonic() - started
                if len(items) != len(indices):
                    raise RuntimeError("SenseVoice result count mismatch")
                output_characters = sum(len(item.text) for item in items)
                max_run = max((maximum_character_run(item.text) for item in items), default=0)
                event_fields = {
                    "worker_id": self.capabilities.worker_id,
                    "batch_id": batch_id,
                    "group_ordinal": group_ordinal,
                    "group_count": len(grouped),
                    "group_size": len(indices),
                    "language": language or "auto",
                    "final_items": sum(1 for index in indices if jobs[index].final),
                    "accumulated_audio_seconds": sum(accumulated),
                    "elapsed_seconds": elapsed,
                    "output_characters": output_characters,
                    "maximum_character_run": max_run,
                }
                events().emit(
                    "asr_engine_group_completed",
                    component="sensevoice_adapter",
                    diagnostic=True,
                    **event_fields,
                )
                if elapsed >= events().slow_engine_seconds:
                    events().emit(
                        "asr_engine_slow_call",
                        component="sensevoice_adapter",
                        level=logging.WARNING,
                        **event_fields,
                    )
                if self._engine_observer is not None:
                    self._engine_observer(
                        group_size=len(indices),
                        group_ordinal=group_ordinal,
                        group_count=len(grouped),
                        elapsed_seconds=elapsed,
                        final=bool(event_fields["final_items"]),
                        accumulated_audio_seconds=sum(accumulated),
                        output_characters=output_characters,
                        maximum_character_run=max_run,
                    )
                for index, item in zip(indices, items):
                    decoded[index] = item

        results: list[InferenceResult] = []
        for state, job, item in zip(states, jobs, decoded):
            if item is None:
                raise RuntimeError("SenseVoice omitted a batch result")
            if state.language is None and item.metadata.get("language") in self.capabilities.languages:
                state.language = item.metadata["language"]
            if job.final:
                state.cached_final = item
            results.append(
                InferenceResult.from_job(
                    job,
                    text=item.text,
                    tail_text=item.text,
                    segment_id=1,
                    final=job.final,
                    metadata=item.metadata or None,
                )
            )
        return results

    async def finish_segment(self, session_id: str) -> AdapterResult:
        state = self._session(session_id)
        item = await self._consume_final(state)
        return AdapterResult(
            backend_session_id=state.backend_session_id,
            text=item.text,
            tail_text=item.text,
            metadata=item.metadata or None,
        )

    async def finish_session(self, session_id: str) -> AdapterResult:
        state = self._session(session_id)
        item = await self._consume_final(state)
        self._sessions.pop(session_id, None)
        return AdapterResult(
            backend_session_id=state.backend_session_id,
            text=item.text,
            tail_text=item.text,
            is_final=True,
            metadata=item.metadata or None,
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
            "session_audio_samples": sum(len(state.pcm) // 2 for state in self._sessions.values()),
        }

    def remaining_segment_samples(self, session_id: str) -> int:
        return max(0, self._max_segment_samples - len(self._session(session_id).pcm) // 2)

    async def _consume_final(self, state: _SessionState) -> SenseVoiceDecoded:
        if state.cached_final is not None:
            item = state.cached_final
        elif state.pcm:
            engine = self._require_ready()
            async with self._engine_lock:
                items = await asyncio.to_thread(
                    engine.transcribe_batch,
                    [_pcm_to_float32(state.pcm)],
                    language=state.language,
                )
            if len(items) != 1:
                raise RuntimeError("SenseVoice final result count mismatch")
            item = items[0]
        else:
            item = SenseVoiceDecoded("", {})
        state.pcm.clear()
        state.cached_final = None
        return item

    def _require_ready(self) -> Any:
        if not self._ready or self._engine is None:
            raise RuntimeError("SenseVoice adapter is not ready")
        return self._engine

    def _session(self, session_id: str) -> _SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError("stale session") from exc


def _pcm_to_float32(pcm: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
