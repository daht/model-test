from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.asr_gateway_chunking import PcmRingBuffer, ReadyChunk
from app.asr_observability import CapacityBufferError
from app.asr_streaming import StreamingTranscriptState


class TerminalState(str, Enum):
    OPEN = "open"
    FINISHING = "finishing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class SessionReservation:
    job_sequence: int
    generation: int
    chunk: ReadyChunk


class GatewaySession:
    def __init__(
        self,
        session_id: str,
        selected_worker_id: str,
        backend_session_id: str,
        *,
        sample_rate: int,
        max_buffer_samples: int,
        language: str = "auto",
        options: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.selected_worker_id = selected_worker_id
        self.backend_session_id = backend_session_id
        self.sample_rate = sample_rate
        self.language = language
        self.options = dict(options or {})
        self.generation = 1
        self.terminal_state = TerminalState.OPEN
        self.buffer = PcmRingBuffer(max_samples=max_buffer_samples)
        self._next_job_sequence = 1
        self._reservation: SessionReservation | None = None
        self._reservation_released = asyncio.Event()
        self._reservation_released.set()
        self.finish_requested = False
        self._accepted_samples = 0
        self._external_discarded_samples = 0
        self.transcript = StreamingTranscriptState(
            sample_rate=sample_rate,
            stable_commit_enabled=False,
            stable_commit_seconds=1,
            stable_commit_min_chars=1,
            stable_commit_min_updates=1,
        )

    @property
    def in_flight(self) -> bool:
        return self._reservation is not None

    @property
    def sample_accounting(self) -> dict[str, int]:
        return {
            "accepted": self._accepted_samples,
            "buffered": self.buffer.buffered_samples,
            "reserved": self.buffer.reserved_samples,
            "acknowledged": self.buffer.acknowledged_samples,
            "discarded": self.buffer.discarded_samples + self._external_discarded_samples,
            "pending_vad": max(
                0,
                self._accepted_samples
                - self.buffer.accepted_samples
                - self._external_discarded_samples,
            ),
        }

    def append_pcm(self, pcm: bytes, *, count_accepted: bool = True) -> tuple[int, int]:
        if self.terminal_state not in {TerminalState.OPEN, TerminalState.FINISHING}:
            raise RuntimeError("session is terminal")
        result = self.buffer.append(pcm)
        if count_accepted:
            self._accepted_samples += len(pcm) // 2
        return result

    def accept_vad_input(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            raise ValueError("pcm_s16le bytes must be sample-aligned")
        samples = len(pcm) // 2
        accounting = self.sample_accounting
        current = sum(
            accounting[key] for key in ("buffered", "reserved", "pending_vad")
        )
        if current + samples > self.buffer.max_samples:
            raise CapacityBufferError(
                "session_pcm_limit",
                limit=self.buffer.max_samples,
                current=current,
                incoming=samples,
                message="session PCM buffer limit exceeded",
            )
        self._accepted_samples += samples

    def record_discarded(self, samples: int) -> None:
        if samples < 0 or samples > self.sample_accounting["pending_vad"]:
            raise ValueError("discarded VAD samples exceed pending input")
        self._external_discarded_samples += samples

    def ready_samples(self, *, preferred: int, maximum: int | None = None, force: bool = False) -> int:
        if self.in_flight:
            return 0
        available = self.buffer.buffered_samples
        maximum = maximum or available
        if available >= maximum > 0:
            return maximum
        if available >= preferred > 0:
            return preferred
        if force or self.finish_requested:
            return available
        return 0

    def reserve(self, samples: int, *, final: bool = False) -> SessionReservation:
        if self.in_flight:
            raise RuntimeError("session already has an in-flight job")
        chunk = self.buffer.reserve_range(samples, final=final)
        reservation = SessionReservation(self._next_job_sequence, self.generation, chunk)
        self._next_job_sequence += 1
        self._reservation = reservation
        self._reservation_released.clear()
        return reservation

    def acknowledge(self, job_sequence: int, *, generation: int) -> ReadyChunk:
        if generation != self.generation:
            raise RuntimeError("stale generation result")
        reservation = self._require_reservation(job_sequence)
        self.buffer.acknowledge(reservation.chunk.start_sample, reservation.chunk.end_sample)
        self._reservation = None
        self._reservation_released.set()
        return reservation.chunk

    def rollback(self, job_sequence: int) -> bool:
        if self._reservation is None or self._reservation.job_sequence != job_sequence:
            return False
        chunk = self._reservation.chunk
        self.buffer.rollback(chunk.start_sample, chunk.end_sample)
        self._reservation = None
        self._reservation_released.set()
        return True

    async def wait_reservation_released(self) -> None:
        await self._reservation_released.wait()

    def matches_reservation(self, job_sequence: int, start_sample: int, end_sample: int) -> bool:
        return bool(
            self._reservation
            and self._reservation.job_sequence == job_sequence
            and self._reservation.generation == self.generation
            and self._reservation.chunk.start_sample == start_sample
            and self._reservation.chunk.end_sample == end_sample
        )

    def request_finish(self) -> None:
        if not self.finish_requested:
            self.finish_requested = True
            self.terminal_state = TerminalState.FINISHING

    def abort(self) -> None:
        if self.terminal_state in {TerminalState.ABORTED, TerminalState.FAILED}:
            return
        if self._reservation:
            chunk = self._reservation.chunk
            self.buffer.rollback(chunk.start_sample, chunk.end_sample)
            self._reservation = None
            self._reservation_released.set()
        if self.buffer.buffered_samples:
            self.buffer.discard(self.buffer.buffered_samples)
        pending = self.sample_accounting["pending_vad"]
        self._external_discarded_samples += pending
        self.generation += 1
        self.terminal_state = TerminalState.ABORTED

    def fail(self) -> None:
        self.abort()
        self.terminal_state = TerminalState.FAILED

    def succeed(self) -> None:
        if self.buffer.buffered_samples or self.buffer.reserved_samples:
            raise RuntimeError("cannot succeed with unaccounted PCM")
        self.terminal_state = TerminalState.SUCCEEDED

    def _require_reservation(self, job_sequence: int) -> SessionReservation:
        if self._reservation is None or self._reservation.job_sequence != job_sequence:
            raise RuntimeError("result does not match in-flight reservation")
        return self._reservation


class SessionManager:
    def __init__(self, *, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions: dict[str, GatewaySession] = {}

    def create(self, session_id: str, worker_id: str, backend_session_id: str, **kwargs: Any) -> GatewaySession:
        if session_id in self._sessions:
            raise ValueError("duplicate session_id")
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError("session capacity exceeded")
        session = GatewaySession(session_id, worker_id, backend_session_id, **kwargs)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> GatewaySession:
        return self._sessions[session_id]

    def close(self, session_id: str) -> GatewaySession | None:
        session = self._sessions.pop(session_id, None)
        if session is not None and session.terminal_state not in {TerminalState.ABORTED, TerminalState.FAILED}:
            session.generation += 1
        return session

    def snapshot(self) -> dict[str, Any]:
        return {"active_sessions": len(self._sessions), "session_ids": sorted(self._sessions)}
