import pytest

from app.asr_gateway_backends import VadMode
from app.asr_gateway_chunking import ChunkPolicy, PcmRingBuffer
from app.asr_observability import CapacityBufferError


def test_ring_buffer_exact_boundary_retains_remainder():
    ring = PcmRingBuffer(max_samples=20)
    ring.append(b"\x01\x00" * 10)
    first = ring.reserve_range(6)
    assert first.start_sample == 0 and first.end_sample == 6
    assert first.pcm == b"\x01\x00" * 6
    ring.acknowledge(first.start_sample, first.end_sample)
    second = ring.reserve_range(4)
    assert second.start_sample == 6 and second.end_sample == 10


def test_ring_buffer_rollback_restores_exact_range():
    ring = PcmRingBuffer(max_samples=10)
    ring.append(b"\x00\x00" * 8)
    reserved = ring.reserve_range(5)
    ring.rollback(reserved.start_sample, reserved.end_sample)
    assert ring.buffered_samples == 8
    again = ring.reserve_range(8)
    assert (again.start_sample, again.end_sample) == (0, 8)


def test_ring_buffer_overflow_has_exact_capacity_reason_and_accounting():
    ring = PcmRingBuffer(max_samples=4)
    ring.append(b"\x00\x00" * 3)

    with pytest.raises(CapacityBufferError, match="buffer") as rejected:
        ring.append(b"\x00\x00" * 2)

    assert rejected.value.reason == "session_pcm_limit"
    assert rejected.value.safe_fields == {"limit": 4, "current": 3, "incoming": 2}


def test_chunk_policy_exact_maximum_and_vad_ownership():
    policy = ChunkPolicy(sample_rate=10, preferred_seconds=0.4, max_seconds=0.6, vad_mode=VadMode.GATEWAY)
    assert policy.preferred_samples == 4
    assert policy.maximum_samples == 6
    assert policy.next_chunk_samples(10, force=False) == 6
    assert policy.next_chunk_samples(3, force=True) == 3
    with pytest.raises(ValueError, match="VAD ownership"):
        ChunkPolicy(sample_rate=10, preferred_seconds=1, max_seconds=2, vad_mode=VadMode.BOTH)
