import pytest

from app.asr_gateway_sessions import GatewaySession, SessionManager


def test_pcm_cursors_conserve_samples_and_rollback():
    session = GatewaySession("s", "w", "b", sample_rate=16_000, max_buffer_samples=20)
    session.append_pcm(b"\x01\x00" * 10)
    reservation = session.reserve(6)
    assert session.sample_accounting == {"accepted": 10, "buffered": 4, "reserved": 6, "acknowledged": 0, "discarded": 0, "pending_vad": 0}
    session.rollback(reservation.job_sequence)
    assert session.sample_accounting["buffered"] == 10
    reservation = session.reserve(6)
    session.acknowledge(reservation.job_sequence, generation=session.generation)
    assert session.sample_accounting["acknowledged"] == 6
    assert sum(session.sample_accounting[k] for k in ("buffered", "reserved", "acknowledged", "discarded")) == 10


def test_session_rejects_alignment_overflow_double_flight_and_stale_result():
    session = GatewaySession("s", "w", "b", sample_rate=16_000, max_buffer_samples=4)
    with pytest.raises(ValueError, match="aligned"):
        session.append_pcm(b"x")
    session.append_pcm(b"\x00\x00" * 4)
    with pytest.raises(BufferError, match="buffer"):
        session.append_pcm(b"\x00\x00")
    reservation = session.reserve(2)
    with pytest.raises(RuntimeError, match="in-flight"):
        session.reserve(1)
    old_generation = session.generation
    session.abort()
    assert session.rollback(reservation.job_sequence) is False
    with pytest.raises(RuntimeError, match="stale generation"):
        session.acknowledge(reservation.job_sequence, generation=old_generation)


def test_finish_flushes_remainder_once_and_audio_coalesces_while_busy():
    session = GatewaySession("s", "w", "b", sample_rate=16_000, max_buffer_samples=20)
    session.append_pcm(b"\x00\x00" * 5)
    first = session.reserve(3)
    session.append_pcm(b"\x00\x00" * 2)
    session.request_finish()
    assert session.ready_samples(preferred=100, force=True) == 0
    session.acknowledge(first.job_sequence, generation=session.generation)
    assert session.ready_samples(preferred=100, force=True) == 4
    final = session.reserve(4, final=True)
    session.acknowledge(final.job_sequence, generation=session.generation)
    assert session.ready_samples(preferred=1, force=True) == 0


def test_session_manager_closes_and_invalidates_generation():
    manager = SessionManager(max_sessions=1)
    session = manager.create("s", "w", "b", sample_rate=16_000, max_buffer_samples=10)
    generation = session.generation
    with pytest.raises(RuntimeError, match="session capacity"):
        manager.create("other", "w", "b", sample_rate=16_000, max_buffer_samples=10)
    assert manager.close("s") is session
    assert session.generation == generation + 1
    assert manager.close("s") is None


def test_abort_explicitly_discards_all_unacknowledged_pcm_once():
    session = GatewaySession("s", "w", "b", sample_rate=16_000, max_buffer_samples=20)
    session.append_pcm(b"\x00\x00" * 10)
    session.reserve(4)
    session.abort()
    accounting = session.sample_accounting
    assert accounting == {"accepted": 10, "buffered": 0, "reserved": 0, "acknowledged": 0, "discarded": 10, "pending_vad": 0}
    session.abort()
    assert session.sample_accounting == accounting
