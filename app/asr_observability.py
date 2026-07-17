from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 1
CAPACITY_REASONS = frozenset(
    {
        "session_pcm_limit",
        "scheduler_ready_job_limit",
        "scheduler_queued_audio_limit",
        "adapter_utterance_limit",
    }
)
SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "secret",
        "token_text",
        "transcript",
        "text",
        "prompt",
        "pcm",
        "audio_path",
        "exception_message",
        "authorization",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class CapacityBufferError(BufferError):
    def __init__(
        self,
        reason: str,
        *,
        limit: int,
        current: int,
        incoming: int,
        message: str | None = None,
    ) -> None:
        if reason not in CAPACITY_REASONS:
            raise ValueError(f"unsupported capacity rejection reason: {reason}")
        for name, value in (
            ("limit", limit),
            ("current", current),
            ("incoming", incoming),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self.reason = reason
        self.safe_fields = {
            "limit": int(limit),
            "current": int(current),
            "incoming": int(incoming),
        }
        super().__init__(message or reason)


def stable_batch_id(jobs: Sequence[Any]) -> str:
    if not jobs:
        raise ValueError("batch identity requires at least one job")
    digest = hashlib.sha256()
    for job in jobs:
        job_id = str(getattr(job, "job_id", ""))
        if not job_id:
            raise ValueError("batch jobs require nonempty job_id")
        digest.update(len(job_id).to_bytes(4, "big"))
        digest.update(job_id.encode("utf-8"))
    return f"batch-{digest.hexdigest()[:16]}"


def maximum_character_run(value: str) -> int:
    maximum = 0
    current = 0
    previous = ""
    for character in value:
        if character == previous:
            current += 1
        else:
            previous = character
            current = 1
        maximum = max(maximum, current)
    return maximum


class BoundedValues:
    def __init__(self, *, maxlen: int = 1024) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("observability values must be finite and nonnegative")
        self._values.append(numeric)

    def summary(self) -> dict[str, float | int]:
        if not self._values:
            return {"count": 0}
        ordered = sorted(self._values)

        def nearest_rank(percentile: float) -> float:
            index = max(0, math.ceil(percentile * len(ordered)) - 1)
            return ordered[index]

        return {
            "count": len(ordered),
            "min": ordered[0],
            "p50": nearest_rank(0.50),
            "p95": nearest_rank(0.95),
            "p99": nearest_rank(0.99),
            "max": ordered[-1],
        }


class EventEmitter:
    def __init__(
        self,
        logger: logging.Logger,
        *,
        diagnostic_enabled: bool = False,
        process_id: str | None = None,
        timestamp: Callable[[], str] = _utc_now,
        slow_engine_seconds: float = 2.0,
    ) -> None:
        if slow_engine_seconds <= 0:
            raise ValueError("slow_engine_seconds must be positive")
        self.logger = logger
        self.diagnostic_enabled = diagnostic_enabled
        self.process_id = process_id or uuid.uuid4().hex
        self.timestamp = timestamp
        self.slow_engine_seconds = float(slow_engine_seconds)

    def emit(
        self,
        event: str,
        *,
        component: str,
        level: int = logging.INFO,
        diagnostic: bool = False,
        **fields: Any,
    ) -> dict[str, Any] | None:
        if diagnostic and not self.diagnostic_enabled:
            return None
        if not event or not component:
            raise ValueError("event and component must be nonempty")
        for name in fields:
            if name.lower() in SENSITIVE_FIELDS:
                raise ValueError(f"sensitive observability field is forbidden: {name}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": self.timestamp(),
            "event": event,
            "level": logging.getLevelName(level),
            "component": component,
            "process_id": self.process_id,
            **fields,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.logger.log(level, encoded)
        return payload


_emitter = EventEmitter(logging.getLogger("app.asr.events"))


def configure_events(
    *, diagnostic_enabled: bool, slow_engine_seconds: float
) -> EventEmitter:
    global _emitter
    logger = logging.getLogger("uvicorn.error.asr.events")
    logger.setLevel(logging.INFO)
    _emitter = EventEmitter(
        logger,
        diagnostic_enabled=diagnostic_enabled,
        slow_engine_seconds=slow_engine_seconds,
    )
    return _emitter


def events() -> EventEmitter:
    return _emitter
