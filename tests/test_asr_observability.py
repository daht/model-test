import json
import logging
from dataclasses import dataclass

import pytest

from app.asr_observability import (
    BoundedValues,
    CapacityBufferError,
    EventEmitter,
    stable_batch_id,
)


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def emitter(*, diagnostic=False):
    logger = logging.getLogger(f"test.asr.events.{diagnostic}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = RecordingHandler()
    logger.addHandler(handler)
    return EventEmitter(
        logger,
        diagnostic_enabled=diagnostic,
        process_id="process-1",
        timestamp=lambda: "2026-07-16T15:30:00.123Z",
    ), handler


def test_event_emitter_writes_stable_json_and_gates_diagnostics():
    events, handler = emitter(diagnostic=False)

    payload = events.emit(
        "asr_session_opened",
        component="gateway",
        session_id="session-1",
        active_sessions=1,
    )
    skipped = events.emit(
        "asr_audio_ingested",
        component="gateway",
        diagnostic=True,
        incoming_samples=3200,
    )

    assert skipped is None
    assert json.loads(handler.messages[0]) == payload == {
        "schema_version": 1,
        "timestamp": "2026-07-16T15:30:00.123Z",
        "event": "asr_session_opened",
        "level": "INFO",
        "component": "gateway",
        "process_id": "process-1",
        "session_id": "session-1",
        "active_sessions": 1,
    }


@pytest.mark.parametrize(
    "field",
    ("api_key", "secret", "token_text", "transcript", "prompt", "pcm", "audio_path", "exception_message"),
)
def test_event_emitter_rejects_sensitive_fields(field):
    events, _ = emitter()

    with pytest.raises(ValueError, match="sensitive observability field"):
        events.emit("asr_bad", component="test", **{field: "value"})


def test_batch_identity_is_ordered_stable_and_contains_no_job_identity():
    @dataclass
    class Job:
        job_id: str

    first = stable_batch_id([Job("session-a:1:1"), Job("session-b:1:1")])
    same = stable_batch_id([Job("session-a:1:1"), Job("session-b:1:1")])
    reversed_id = stable_batch_id([Job("session-b:1:1"), Job("session-a:1:1")])

    assert first == same
    assert first != reversed_id
    assert "session" not in first


def test_capacity_error_keeps_buffererror_compatibility_and_safe_fields():
    error = CapacityBufferError(
        "session_pcm_limit", limit=96_000, current=90_000, incoming=6_400
    )

    assert isinstance(error, BufferError)
    assert error.reason == "session_pcm_limit"
    assert error.safe_fields == {
        "limit": 96_000,
        "current": 90_000,
        "incoming": 6_400,
    }


def test_bounded_values_reports_nearest_rank_percentiles_and_rolls_window():
    values = BoundedValues(maxlen=3)
    for value in (1, 2, 3, 4):
        values.add(value)

    assert values.summary() == {
        "count": 3,
        "min": 2.0,
        "p50": 3.0,
        "p95": 4.0,
        "p99": 4.0,
        "max": 4.0,
    }
    assert BoundedValues(maxlen=2).summary() == {"count": 0}
