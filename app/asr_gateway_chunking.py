from __future__ import annotations

from dataclasses import dataclass

from app.asr_gateway_backends import VadMode


@dataclass(frozen=True)
class ReadyChunk:
    start_sample: int
    end_sample: int
    pcm: bytes
    final: bool = False

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


class PcmRingBuffer:
    """Aligned PCM buffer with absolute cursors and one explicit reservation."""

    def __init__(self, *, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self._data = bytearray()
        self._base_sample = 0
        self.accepted_samples = 0
        self.acknowledged_samples = 0
        self.discarded_samples = 0
        self._reservation: ReadyChunk | None = None

    @property
    def reserved_samples(self) -> int:
        return self._reservation.sample_count if self._reservation else 0

    @property
    def buffered_samples(self) -> int:
        return len(self._data) // 2 - self.reserved_samples

    @property
    def next_unreserved_sample(self) -> int:
        return self._base_sample + self.reserved_samples

    def append(self, pcm: bytes) -> tuple[int, int]:
        if len(pcm) % 2:
            raise ValueError("pcm_s16le bytes must be sample-aligned")
        samples = len(pcm) // 2
        if len(self._data) // 2 + samples > self.max_samples:
            raise BufferError("session PCM buffer limit exceeded")
        start = self.accepted_samples
        self._data.extend(pcm)
        self.accepted_samples += samples
        return start, self.accepted_samples

    def reserve_range(self, samples: int, *, final: bool = False) -> ReadyChunk:
        if self._reservation is not None:
            raise RuntimeError("one in-flight reservation is already active")
        if samples <= 0 or samples > self.buffered_samples:
            raise ValueError("reservation sample count is outside buffered range")
        start = self._base_sample
        end = start + samples
        chunk = ReadyChunk(start, end, bytes(self._data[: samples * 2]), final)
        self._reservation = chunk
        return chunk

    def acknowledge(self, start_sample: int, end_sample: int) -> None:
        reservation = self._require_reservation(start_sample, end_sample)
        samples = reservation.sample_count
        del self._data[: samples * 2]
        self._base_sample = reservation.end_sample
        self.acknowledged_samples += samples
        self._reservation = None

    def rollback(self, start_sample: int, end_sample: int) -> None:
        self._require_reservation(start_sample, end_sample)
        self._reservation = None

    def discard(self, samples: int) -> None:
        if self._reservation is not None:
            raise RuntimeError("cannot discard while a reservation is in-flight")
        if samples < 0 or samples > self.buffered_samples:
            raise ValueError("discard sample count is outside buffered range")
        del self._data[: samples * 2]
        self._base_sample += samples
        self.discarded_samples += samples

    def _require_reservation(self, start: int, end: int) -> ReadyChunk:
        if self._reservation is None or (self._reservation.start_sample, self._reservation.end_sample) != (start, end):
            raise RuntimeError("reservation range does not match active reservation")
        return self._reservation


@dataclass(frozen=True)
class ChunkPolicy:
    sample_rate: int
    preferred_seconds: float
    max_seconds: float
    vad_mode: VadMode = VadMode.NONE
    minimum_seconds: float = 0.0
    overlap_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.preferred_seconds <= 0 or self.max_seconds <= 0:
            raise ValueError("chunk durations must be positive")
        if self.preferred_seconds > self.max_seconds:
            raise ValueError("preferred duration exceeds maximum")
        if self.vad_mode is VadMode.BOTH:
            raise ValueError("VAD ownership cannot be both gateway and worker")

    @property
    def preferred_samples(self) -> int:
        return round(self.preferred_seconds * self.sample_rate)

    @property
    def maximum_samples(self) -> int:
        return round(self.max_seconds * self.sample_rate)

    @property
    def minimum_samples(self) -> int:
        return round(self.minimum_seconds * self.sample_rate)

    def next_chunk_samples(self, buffered_samples: int, *, force: bool = False) -> int:
        if buffered_samples >= self.maximum_samples:
            return self.maximum_samples
        if buffered_samples >= self.preferred_samples:
            return self.preferred_samples
        if force and buffered_samples >= self.minimum_samples:
            return buffered_samples
        return 0
