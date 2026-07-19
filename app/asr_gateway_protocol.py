from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from app.asr_gateway_backends import ResultMode
from app.asr_streaming import StreamingTranscriptState, TranscriptEvent


@dataclass(frozen=True)
class StartCommand:
    type: str = "start"
    format: str = "pcm_s16le"
    sample_rate: int = 16_000
    channels: int = 1
    language: str = "auto"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlCommand:
    type: str


def parse_client_command(payload: Mapping[str, Any]) -> StartCommand | ControlCommand:
    if not isinstance(payload, Mapping):
        raise ValueError("command must be an object")
    command_type = payload.get("type")
    if command_type == "start":
        if payload.get("format", "pcm_s16le") != "pcm_s16le":
            raise ValueError("format must be pcm_s16le")
        if payload.get("sample_rate", 16_000) != 16_000:
            raise ValueError("sample_rate must be 16000")
        if payload.get("channels", 1) != 1:
            raise ValueError("channels must be one")
        language = str(payload.get("language", "auto")).strip().lower()
        if not language or len(language) > 32:
            raise ValueError("language is invalid")
        options = payload.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("options must be an object")
        allowed = {"timestamps", "task", "prompt", "hotwords"}
        if set(options) - allowed:
            raise ValueError("options contain unsupported fields")
        return StartCommand(language=language, options=dict(options))
    if command_type in {"segment", "finish", "abort"}:
        return ControlCommand(str(command_type))
    raise ValueError("unsupported command type")


class ProtocolSession:
    def __init__(self, *, sample_rate: int, segment_local: bool = False) -> None:
        self.state = StreamingTranscriptState(
            sample_rate=sample_rate,
            stable_commit_enabled=False,
            stable_commit_seconds=1,
            stable_commit_min_chars=1,
            stable_commit_min_updates=1,
            segment_local_snapshots=segment_local,
        )
        self.terminal = False

    def ready(self, **metadata: Any) -> dict[str, Any]:
        self._require_open()
        return self._serialize(self.state.new_event("ready"), **metadata)

    def apply_result(
        self,
        mode: ResultMode,
        *,
        text: str = "",
        confirmed_text: str = "",
        tail_text: str = "",
        decoded_samples: int = 0,
        segment_id: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._require_open()
        if mode is ResultMode.CUMULATIVE_SNAPSHOT:
            events = self.state.apply_model_update(text, processed_samples=decoded_samples)
        elif mode is ResultMode.REPLACEABLE_SEGMENT:
            events = self.state.apply_segment_snapshot(segment_id, text, decoded_samples_delta=decoded_samples)
        elif mode is ResultMode.CONFIRMED_PLUS_TAIL:
            if confirmed_text:
                current = self.state.confirmed_text + confirmed_text
                self.state.apply_model_update(current, processed_samples=decoded_samples)
                events = self.state.commit_pending()
                decoded_samples = 0
            else:
                events = []
            events.extend(
                self.state.apply_model_update(
                    self.state.confirmed_text + tail_text,
                    processed_samples=decoded_samples,
                )
            )
        else:
            raise ValueError("unsupported result mode")
        extra = {"metadata": dict(metadata)} if metadata is not None else {}
        return [self._serialize(event, **extra) for event in events]

    def segment(self, *, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        self._require_open()
        extra = {"metadata": dict(metadata)} if metadata is not None else {}
        return [self._serialize(event, **extra) for event in self.state.commit_pending()]

    def final(
        self,
        tail_text: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        if tail_text is not None:
            self.state.apply_model_update(self.state.confirmed_text + tail_text, processed_samples=0)
        extra = {"metadata": dict(metadata)} if metadata is not None else {}
        event = self._serialize(self.state.final_event(), **extra)
        self.terminal = True
        return event

    def error(self, exc: BaseException, *, code: str = "backend_error") -> dict[str, Any]:
        self._require_open()
        event = self._serialize(
            self.state.new_event("error"),
            code=code,
            message=f"{type(exc).__name__}: operation failed",
        )
        self.terminal = True
        return event

    def _require_open(self) -> None:
        if self.terminal:
            raise RuntimeError("protocol session is terminal")

    @staticmethod
    def _serialize(event: TranscriptEvent, **extra: Any) -> dict[str, Any]:
        return {"type": event.type, "text": event.text, "sequence": event.sequence, **extra}
