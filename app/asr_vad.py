from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class VADEndpointState(str, Enum):
    WAITING_FOR_SPEECH = "waiting_for_speech"
    SPEECH_CANDIDATE = "speech_candidate"
    IN_SPEECH = "in_speech"
    TRAILING_SILENCE = "trailing_silence"
    FINALIZING = "finalizing"


class VADRuntimeError(RuntimeError):
    pass


class VADBackend(Protocol):
    def speech_probability(self, pcm_s16le: bytes) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class VADDecision:
    audio_to_model: bytes = b""
    endpoint: bool = False
    discarded_samples: int = 0
    transitions: tuple[tuple[VADEndpointState, VADEndpointState], ...] = ()


class EnergyVADBackend:
    """Deterministic test substitute used only by the explicit mock backend."""

    def __init__(self, rms_threshold: int) -> None:
        self.rms_threshold = rms_threshold

    def speech_probability(self, pcm_s16le: bytes) -> float:
        sample_count = len(pcm_s16le) // 2
        if not sample_count:
            return 0.0
        total_square = 0
        for index in range(0, sample_count * 2, 2):
            sample = int.from_bytes(
                pcm_s16le[index : index + 2], "little", signed=True
            )
            total_square += sample * sample
        rms = (total_square / sample_count) ** 0.5
        return 1.0 if rms > self.rms_threshold else 0.0

    def reset(self) -> None:
        return None


class SileroONNXVADBackend:
    """Per-stream Silero recurrent state backed by ONNX Runtime on CPU."""

    def __init__(
        self,
        model_path: str,
        expected_sha256: str,
        *,
        sample_rate: int = 16000,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ) -> None:
        if sample_rate != 16000:
            raise VADRuntimeError("Silero VAD requires 16000 Hz audio")
        path = Path(model_path)
        if not path.is_file():
            raise VADRuntimeError(f"Silero VAD model asset is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise VADRuntimeError("Silero VAD model checksum mismatch")
        try:
            import numpy as np
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise VADRuntimeError(
                "Silero VAD requires pinned numpy and onnxruntime dependencies"
            ) from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise VADRuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self._session = ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise VADRuntimeError("Silero VAD ONNX model failed to initialize") from exc
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise VADRuntimeError("Silero VAD must run only on CPUExecutionProvider")

        self._np = np
        self._sample_rate = sample_rate
        self._context_samples = 64
        self.reset()

    def speech_probability(self, pcm_s16le: bytes) -> float:
        np = self._np
        audio = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32)
        if audio.size != 512:
            raise ValueError("Silero VAD frames must contain exactly 512 samples")
        audio /= 32768.0
        audio = audio.reshape(1, -1)
        model_input = np.concatenate((self._context, audio), axis=1)
        outputs = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(self._sample_rate, dtype=np.int64),
            },
        )
        self._state = outputs[1]
        self._context = model_input[:, -self._context_samples :]
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def reset(self) -> None:
        if not hasattr(self, "_np"):
            return
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros(
            (1, self._context_samples), dtype=self._np.float32
        )


class StreamingVADEndpointDetector:
    def __init__(
        self,
        *,
        backend: VADBackend,
        sample_rate: int,
        frame_samples: int,
        onset_threshold: float,
        offset_threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        hangover_ms: int,
        pre_roll_ms: int,
    ) -> None:
        self.backend = backend
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * 2
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_speech_samples = self._milliseconds_to_samples(min_speech_ms)
        self.min_silence_samples = self._milliseconds_to_samples(min_silence_ms)
        self.hangover_samples = self._milliseconds_to_samples(hangover_ms)
        self.pre_roll_samples = self._milliseconds_to_samples(pre_roll_ms)
        self.state = VADEndpointState.WAITING_FOR_SPEECH
        self._input_buffer = bytearray()
        self._pre_roll = bytearray()
        self._candidate = bytearray()
        self._candidate_speech_samples = 0
        self._trailing = bytearray()

    def add_audio(self, pcm_s16le: bytes) -> VADDecision:
        if len(pcm_s16le) % 2:
            raise ValueError("VAD audio must be aligned pcm_s16le")
        self._input_buffer.extend(pcm_s16le)
        return self._drain_complete_frames()

    def endpoint_finalized(self) -> VADDecision:
        if self.state is not VADEndpointState.FINALIZING:
            return VADDecision()
        transitions = [self._transition(VADEndpointState.WAITING_FOR_SPEECH)]
        self.backend.reset()
        return self._drain_complete_frames(initial_transitions=transitions)

    def reset(self) -> None:
        self.state = VADEndpointState.WAITING_FOR_SPEECH
        self._input_buffer.clear()
        self._pre_roll.clear()
        self._candidate.clear()
        self._candidate_speech_samples = 0
        self._trailing.clear()
        self.backend.reset()

    def finish_input(self) -> VADDecision:
        decision = self._drain_complete_frames()
        audio = bytearray(decision.audio_to_model)
        discarded = decision.discarded_samples
        transitions = list(decision.transitions)
        if self.state is VADEndpointState.IN_SPEECH:
            audio.extend(self._input_buffer)
        elif self.state is VADEndpointState.TRAILING_SILENCE:
            remaining_hangover = max(
                0, self.hangover_samples - len(audio) // 2
            )
            keep_bytes = min(len(self._trailing), remaining_hangover * 2)
            audio.extend(self._trailing[:keep_bytes])
            discarded += (len(self._trailing) - keep_bytes) // 2
            discarded += len(self._input_buffer) // 2
        else:
            discarded += len(self._input_buffer) // 2
            if self.state is VADEndpointState.SPEECH_CANDIDATE:
                discarded += len(self._candidate) // 2
                transitions.append(
                    self._transition(VADEndpointState.WAITING_FOR_SPEECH)
                )
        discarded += len(self._pre_roll) // 2
        self._input_buffer.clear()
        self._pre_roll.clear()
        self._candidate.clear()
        self._candidate_speech_samples = 0
        self._trailing.clear()
        return VADDecision(bytes(audio), False, discarded, tuple(transitions))

    def _drain_complete_frames(
        self,
        *,
        initial_transitions: list[
            tuple[VADEndpointState, VADEndpointState]
        ] | None = None,
    ) -> VADDecision:
        audio = bytearray()
        discarded = 0
        transitions = list(initial_transitions or ())
        while (
            self.state is not VADEndpointState.FINALIZING
            and len(self._input_buffer) >= self.frame_bytes
        ):
            frame = bytes(self._input_buffer[: self.frame_bytes])
            del self._input_buffer[: self.frame_bytes]
            probability = self.backend.speech_probability(frame)
            frame_audio, endpoint, frame_discarded, transition = self._process_frame(
                frame, probability
            )
            audio.extend(frame_audio)
            discarded += frame_discarded
            transitions.extend(transition)
            if endpoint:
                return VADDecision(bytes(audio), True, discarded, tuple(transitions))
        return VADDecision(bytes(audio), False, discarded, tuple(transitions))

    def _process_frame(
        self, frame: bytes, probability: float
    ) -> tuple[
        bytes,
        bool,
        int,
        tuple[tuple[VADEndpointState, VADEndpointState], ...],
    ]:
        transitions: list[tuple[VADEndpointState, VADEndpointState]] = []
        if self.state is VADEndpointState.WAITING_FOR_SPEECH:
            if probability >= self.onset_threshold:
                self._candidate = bytearray(self._pre_roll)
                self._pre_roll.clear()
                self._candidate.extend(frame)
                self._candidate_speech_samples = self.frame_samples
                transitions.append(self._transition(VADEndpointState.SPEECH_CANDIDATE))
                if self._candidate_speech_samples >= self.min_speech_samples:
                    transitions.append(self._transition(VADEndpointState.IN_SPEECH))
                    audio = bytes(self._candidate)
                    self._candidate.clear()
                    self._candidate_speech_samples = 0
                    return audio, False, 0, tuple(transitions)
            else:
                discarded = self._append_pre_roll(frame)
                return b"", False, discarded, tuple(transitions)
            return b"", False, 0, tuple(transitions)

        if self.state is VADEndpointState.SPEECH_CANDIDATE:
            if probability >= self.offset_threshold:
                self._candidate.extend(frame)
                self._candidate_speech_samples += self.frame_samples
                if self._candidate_speech_samples >= self.min_speech_samples:
                    transitions.append(self._transition(VADEndpointState.IN_SPEECH))
                    audio = bytes(self._candidate)
                    self._candidate.clear()
                    self._candidate_speech_samples = 0
                    return audio, False, 0, tuple(transitions)
                return b"", False, 0, tuple(transitions)
            discarded = self._append_pre_roll(bytes(self._candidate) + frame)
            self._candidate.clear()
            self._candidate_speech_samples = 0
            transitions.append(self._transition(VADEndpointState.WAITING_FOR_SPEECH))
            logger.info("asr_vad_discarded category=short_burst samples=%d", discarded)
            return b"", False, discarded, tuple(transitions)

        if self.state is VADEndpointState.IN_SPEECH:
            if probability >= self.offset_threshold:
                return frame, False, 0, ()
            self._trailing = bytearray(frame)
            transitions.append(self._transition(VADEndpointState.TRAILING_SILENCE))

        elif self.state is VADEndpointState.TRAILING_SILENCE:
            if probability >= self.offset_threshold:
                audio = bytes(self._trailing) + frame
                self._trailing.clear()
                transitions.append(self._transition(VADEndpointState.IN_SPEECH))
                return audio, False, 0, tuple(transitions)
            self._trailing.extend(frame)

        if len(self._trailing) // 2 < self.min_silence_samples:
            return b"", False, 0, tuple(transitions)

        trailing = bytes(self._trailing)
        keep_bytes = min(len(trailing), self.hangover_samples * 2)
        audio = trailing[:keep_bytes]
        self._pre_roll.clear()
        discarded = self._append_pre_roll(trailing[keep_bytes:])
        self._trailing.clear()
        transitions.append(self._transition(VADEndpointState.FINALIZING))
        logger.info(
            "asr_vad_endpoint hangover_samples=%d discarded_samples=%d",
            len(audio) // 2,
            discarded,
        )
        return audio, True, discarded, tuple(transitions)

    def _append_pre_roll(self, audio: bytes) -> int:
        self._pre_roll.extend(audio)
        max_bytes = self.pre_roll_samples * 2
        discarded_bytes = max(0, len(self._pre_roll) - max_bytes)
        if len(self._pre_roll) > max_bytes:
            del self._pre_roll[:discarded_bytes]
        return discarded_bytes // 2

    def _transition(
        self, target: VADEndpointState
    ) -> tuple[VADEndpointState, VADEndpointState]:
        source = self.state
        self.state = target
        logger.info("asr_vad_transition from_state=%s to_state=%s", source.value, target.value)
        return source, target

    def _milliseconds_to_samples(self, milliseconds: int) -> int:
        return max(1, round(self.sample_rate * milliseconds / 1000))


def create_vad_endpoint_detector(settings: Settings) -> StreamingVADEndpointDetector:
    if settings.asr_backend == "mock":
        backend: VADBackend = EnergyVADBackend(settings.asr_vad_rms_threshold)
    else:
        backend = SileroONNXVADBackend(
            settings.asr_vad_model_path,
            settings.asr_vad_model_sha256,
            intra_op_threads=settings.asr_vad_onnx_intra_threads,
            inter_op_threads=settings.asr_vad_onnx_inter_threads,
        )
    return StreamingVADEndpointDetector(
        backend=backend,
        sample_rate=16000,
        frame_samples=settings.asr_vad_frame_samples,
        onset_threshold=settings.asr_vad_onset_threshold,
        offset_threshold=settings.asr_vad_offset_threshold,
        min_speech_ms=settings.asr_vad_min_speech_ms,
        min_silence_ms=settings.asr_vad_min_silence_ms,
        hangover_ms=settings.asr_vad_hangover_ms,
        pre_roll_ms=settings.asr_vad_pre_roll_ms,
    )
