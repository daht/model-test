from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.asr import ASRTranscriber, create_asr_transcriber
from app.audio import remove_file, save_upload_to_tempfile, write_pcm_s16le_wav
from app.auth import is_valid_api_key, require_api_key
from app.config import Settings, get_settings
from app.asr_inference import (
    ASRBatchConflict,
    ASRFileTranscriptionDisabled,
    ASRInferenceCoordinator,
    ASRInferenceTimeout,
    ASRNotReady,
    ASRQueueFull,
    ASRQueueTimeout,
    ASRSessionLimit,
    ASRSessionPoisoned,
)
from app.asr_streaming import ConfirmedPrefixConflict, StreamingTranscriptState
from app.schemas import (
    ASRHealthResponse,
    ASRReadyResponse,
    TranscribeResponse,
    TranscribeStreamInfoResponse,
)

settings = get_settings()
asr_transcriber = create_asr_transcriber(settings)
asr_coordinator = ASRInferenceCoordinator(settings, lambda: create_asr_transcriber(settings))

SENTENCE_TERMINATORS = set("。？！｡?!؟۔।॥။.")
SENTENCE_CLOSERS = set("\"'”’」』）)】]》〉")
DOT_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "st.",
    "jr.",
    "sr.",
    "ph.d.",
}

@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asr_coordinator.start()
    try:
        yield
    finally:
        await asr_coordinator.stop()


app = FastAPI(
    title="Qwen ASR REST API",
    version="0.1.0",
    description="REST and WebSocket API template for Qwen3-ASR-1.7B deployment.",
    lifespan=lifespan,
)


def get_asr_transcriber() -> ASRTranscriber:
    return asr_transcriber


def get_asr_coordinator() -> ASRInferenceCoordinator:
    return asr_coordinator


@dataclass
class SentenceCommitter:
    commit_on_punctuation: bool = False
    pending_text: str = ""
    confirmed_texts: list[str] = field(default_factory=list)

    def append(self, text: str) -> list[str]:
        if not text:
            return []

        self.pending_text = self._merge_recognition_text(text)
        return self._commit_complete_sentences()

    def append_cumulative(self, text: str) -> list[str]:
        self.pending_text = self._tail_after_confirmed_text(text)
        return self._commit_complete_sentences()

    def _commit_complete_sentences(self) -> list[str]:
        if not self.commit_on_punctuation:
            return []

        committed, self.pending_text = _split_committed_sentences(self.pending_text)
        self.confirmed_texts.extend(committed)
        return committed

    def _merge_recognition_text(self, text: str) -> str:
        confirmed_text = "".join(self.confirmed_texts)
        current_text = confirmed_text + self.pending_text

        if confirmed_text and text.startswith(confirmed_text):
            return text[len(confirmed_text) :]

        if current_text and text.startswith(current_text):
            return text[len(confirmed_text) :]

        if self.pending_text and text.startswith(self.pending_text):
            return text

        return self.pending_text + text

    def _tail_after_confirmed_text(self, text: str) -> str:
        confirmed_text = "".join(self.confirmed_texts)
        if not confirmed_text:
            return text

        if text.startswith(confirmed_text):
            return text[len(confirmed_text) :]

        overlap = min(len(confirmed_text), len(text))
        while overlap > 0:
            if confirmed_text.endswith(text[:overlap]):
                return text[overlap:]
            overlap -= 1

        return text

    def commit_pending(self) -> str:
        sentence = self.pending_text.rstrip()
        if sentence:
            self.confirmed_texts.append(sentence)
        self.pending_text = ""
        return sentence

    def commit_prefix(self, prefix: str) -> str:
        if not prefix or not self.pending_text.startswith(prefix):
            return ""
        self.confirmed_texts.append(prefix)
        self.pending_text = self.pending_text[len(prefix) :]
        return prefix


@dataclass
class StablePunctuationCommitter:
    enabled: bool
    stable_seconds: float
    min_chars: int
    min_updates: int
    candidate: str = ""
    candidate_since: float = 0.0
    candidate_updates: int = 0

    def observe(self, text: str, now: float) -> str | None:
        next_candidate = _first_stable_punctuation_candidate(text, self.min_chars) if self.enabled else ""
        if not next_candidate:
            self.reset()
            return None

        if next_candidate != self.candidate:
            self.candidate = next_candidate
            self.candidate_since = now
            self.candidate_updates = 1
            return None

        self.candidate_updates += 1
        if self.candidate_updates < self.min_updates:
            return None
        if now - self.candidate_since < self.stable_seconds:
            return None

        committed = self.candidate
        self.reset()
        return committed

    def reset(self) -> None:
        self.candidate = ""
        self.candidate_since = 0.0
        self.candidate_updates = 0


@dataclass
class SilenceEndpointDetector:
    silence_seconds: float
    rms_threshold: int
    current_silence_seconds: float = 0.0
    committed_for_current_silence: bool = False

    def add_audio(self, pcm_bytes: bytes, sample_rate: int) -> bool:
        if not pcm_bytes:
            return False

        if _pcm_s16le_rms(pcm_bytes) > self.rms_threshold:
            self.current_silence_seconds = 0.0
            self.committed_for_current_silence = False
            return False

        self.current_silence_seconds += _pcm_s16le_duration_seconds(pcm_bytes, sample_rate)
        if self.current_silence_seconds < self.silence_seconds or self.committed_for_current_silence:
            return False

        self.committed_for_current_silence = True
        return True

    def reset(self) -> None:
        self.current_silence_seconds = 0.0
        self.committed_for_current_silence = False


def _pcm_s16le_rms(pcm_bytes: bytes) -> int:
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return 0

    total_square = 0
    for index in range(0, sample_count * 2, 2):
        sample = int.from_bytes(pcm_bytes[index : index + 2], byteorder="little", signed=True)
        total_square += sample * sample

    return int((total_square / sample_count) ** 0.5)


def _pcm_s16le_duration_seconds(pcm_bytes: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (len(pcm_bytes) // 2) / sample_rate


def _split_committed_sentences(text: str) -> tuple[list[str], str]:
    committed: list[str] = []
    sentence_start = 0
    index = 0

    while index < len(text):
        char = text[index]
        if char in SENTENCE_TERMINATORS and _is_sentence_end(text, index):
            sentence_end = _sentence_boundary_end(text, index)

            sentence = text[sentence_start:sentence_end].strip()
            if sentence:
                committed.append(sentence)

            sentence_start = sentence_end
            while sentence_start < len(text) and text[sentence_start].isspace():
                sentence_start += 1
            index = sentence_start
            continue

        index += 1

    return committed, text[sentence_start:]


def _first_stable_punctuation_candidate(text: str, min_chars: int) -> str:
    index = 0
    while index < len(text):
        if text[index] in SENTENCE_TERMINATORS and _is_sentence_end(text, index):
            sentence_end = _sentence_boundary_end(text, index)
            candidate = text[:sentence_end]
            if len("".join(candidate.split())) >= min_chars:
                return candidate
            index = sentence_end
            continue
        index += 1
    return ""


def _sentence_boundary_end(text: str, index: int) -> int:
    sentence_end = index + 1
    while sentence_end < len(text):
        char = text[sentence_end]
        if char not in SENTENCE_TERMINATORS and char not in SENTENCE_CLOSERS:
            break
        sentence_end += 1
    return sentence_end


def _is_sentence_end(text: str, index: int) -> bool:
    if text[index] != ".":
        return True

    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""

    if previous_char.isdigit() and next_char.isdigit():
        return False

    if previous_char.isalnum() and next_char.isalnum():
        return False

    dot_token = _dot_token(text, index)

    if dot_token.lower() in DOT_ABBREVIATIONS:
        return False

    if len(dot_token) == 2 and dot_token[0].isupper():
        return False

    if _is_dotted_initialism(dot_token):
        return False

    return True


def _is_dotted_initialism(token: str) -> bool:
    parts = token.split(".")
    initials = [part for part in parts if part]
    return len(initials) >= 2 and all(len(part) == 1 and part.isupper() for part in initials)


def _dot_token(text: str, index: int) -> str:
    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start : index + 1]


@app.get("/health", response_model=ASRHealthResponse)
def health(current_settings: Settings = Depends(get_settings)) -> ASRHealthResponse:
    return ASRHealthResponse(
        status="ok",
        model=current_settings.asr_model_name,
        backend=current_settings.asr_backend,
    )


@app.get("/ready", response_model=ASRReadyResponse)
def ready(
    current_settings: Settings = Depends(get_settings),
    current_coordinator: ASRInferenceCoordinator = Depends(get_asr_coordinator),
):
    snapshot = current_coordinator.snapshot()
    payload = ASRReadyResponse(
        status="ready" if snapshot.ready and snapshot.accepting else "not_ready",
        model=current_settings.asr_model_name,
        backend=current_settings.asr_backend,
        active_streams=snapshot.active_streams,
        queue_depth=snapshot.queue_depth,
        queued_audio_seconds=snapshot.queued_audio_seconds,
        detail=snapshot.load_error,
    )
    if not snapshot.ready or not snapshot.accepting:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())
    return payload


@app.post(
    "/v1/transcribe",
    response_model=TranscribeResponse,
    dependencies=[Depends(require_api_key)],
)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    current_settings: Settings = Depends(get_settings),
    current_coordinator: ASRInferenceCoordinator = Depends(get_asr_coordinator),
) -> TranscribeResponse:
    temp_path = await save_upload_to_tempfile(
        file,
        max_bytes=current_settings.asr_max_upload_mb * 1024 * 1024,
    )
    try:
        result = await current_coordinator.transcribe_file(temp_path, language)
    except ASRFileTranscriptionDisabled as exc:
        raise _http_asr_unavailable("file_transcription_disabled", str(exc)) from exc
    except ASRBatchConflict as exc:
        raise _http_asr_unavailable("batch_conflict", str(exc)) from exc
    except (ASRQueueFull, ASRQueueTimeout) as exc:
        raise _http_asr_unavailable("server_busy", str(exc)) from exc
    except ASRInferenceTimeout as exc:
        raise _http_asr_unavailable("inference_timeout", str(exc)) from exc
    except (ASRNotReady, ASRSessionLimit) as exc:
        raise _http_asr_unavailable("not_ready", str(exc)) from exc
    finally:
        remove_file(temp_path)

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        model=current_settings.asr_model_name,
    )


@app.get("/v1/transcribe/stream-info", response_model=TranscribeStreamInfoResponse)
def transcribe_stream_info(current_settings: Settings = Depends(get_settings)) -> TranscribeStreamInfoResponse:
    return TranscribeStreamInfoResponse(
        protocol_version=current_settings.asr_protocol_version,
        file_transcribe_enabled=current_settings.asr_file_transcribe_enabled,
        websocket_url="/v1/transcribe/stream",
        audio_format={
            "format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "recommended_chunk_ms": "100-500",
            "backend": current_settings.asr_backend,
            "stream_mode": current_settings.asr_stream_mode,
            "vad_silence_seconds": current_settings.asr_vad_silence_seconds,
            "commit_on_punctuation": current_settings.asr_commit_on_punctuation,
            "stateful": {
                "chunk_seconds": current_settings.asr_stream_chunk_seconds,
                "unfixed_chunk_num": current_settings.asr_stream_unfixed_chunk_num,
                "unfixed_token_num": current_settings.asr_stream_unfixed_token_num,
                "vllm_gpu_memory_utilization": current_settings.asr_vllm_gpu_memory_utilization,
                "vllm_max_new_tokens": current_settings.asr_vllm_max_new_tokens,
                "stable_commit_enabled": current_settings.asr_stable_commit_enabled,
                "stable_commit_seconds": current_settings.asr_stable_commit_seconds,
                "stable_commit_min_chars": current_settings.asr_stable_commit_min_chars,
                "stable_commit_min_updates": current_settings.asr_stable_commit_min_updates,
            },
        },
        start_message={
            "type": "start",
            "api_key": "<your-api-key>",
            "language": "zh",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        },
        end_message={"type": "end"},
        segment_message={"type": "segment"},
        server_messages=[
            {"type": "ready", "sequence": 1},
            {"type": "partial", "text": "...", "sequence": 2},
            {"type": "sentence_final", "text": "...", "sequence": 3},
            {"type": "final", "text": "...", "sequence": 4},
            {"type": "error", "code": "...", "message": "...", "sequence": 5},
        ],
    )


def _http_asr_unavailable(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": code, "message": message},
    )


class ASRStreamStart(BaseModel):
    type: Literal["start"]
    api_key: str
    language: str | None = Field(default=None, max_length=32)
    sample_rate: int = Field(default=16000, strict=True)
    format: Literal["pcm_s16le"] = "pcm_s16le"

    model_config = ConfigDict(extra="ignore")


class _StreamClosed(RuntimeError):
    pass


class StreamingSessionController:
    def __init__(
        self,
        websocket: WebSocket,
        current_settings: Settings,
        coordinator: ASRInferenceCoordinator,
    ) -> None:
        self.websocket = websocket
        self.settings = current_settings
        self.coordinator = coordinator
        self.session_id: str | None = None
        self.finished = False
        self.accepted_samples = 0
        self.transcript = StreamingTranscriptState(
            sample_rate=16000,
            stable_commit_enabled=current_settings.asr_stable_commit_enabled,
            stable_commit_seconds=current_settings.asr_stable_commit_seconds,
            stable_commit_min_chars=current_settings.asr_stable_commit_min_chars,
            stable_commit_min_updates=current_settings.asr_stable_commit_min_updates,
        )
        self.silence = SilenceEndpointDetector(
            silence_seconds=current_settings.asr_vad_silence_seconds,
            rms_threshold=current_settings.asr_vad_rms_threshold,
        )

    async def start(self, language: str | None) -> None:
        try:
            self.session_id = await self.coordinator.create_stream(language)
        except ValueError as exc:
            await self.fail("invalid_language", "Unsupported language", 1003)
            raise _StreamClosed from exc
        except (ASRQueueFull, ASRQueueTimeout, ASRSessionLimit, ASRBatchConflict) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRNotReady as exc:
            await self.fail("not_ready", "ASR is not ready", 1013)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "Unable to create ASR session", 1011)
            raise _StreamClosed from exc
        await self._send_event(self.transcript.new_event("ready"))

    async def add_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or len(pcm_bytes) % 2:
            await self.fail("invalid_audio_frame", "Audio frames must contain aligned PCM samples", 1003)
            raise _StreamClosed
        if len(pcm_bytes) > self.settings.asr_max_frame_bytes:
            await self.fail("frame_too_large", "Audio frame exceeds the configured limit", 1009)
            raise _StreamClosed
        sample_count = len(pcm_bytes) // 2
        max_samples = int(self.settings.asr_max_audio_seconds * 16000)
        if self.accepted_samples + sample_count > max_samples:
            await self.fail("audio_limit_exceeded", "Maximum audio duration exceeded", 1008)
            raise _StreamClosed
        self.accepted_samples += sample_count
        assert self.session_id is not None
        try:
            result = await self.coordinator.add_audio(self.session_id, pcm_bytes, 16000)
        except (ASRQueueFull, ASRQueueTimeout, ASRSessionLimit, ASRBatchConflict) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRInferenceTimeout as exc:
            await self.fail("inference_timeout", "ASR inference timed out", 1011)
            raise _StreamClosed from exc
        except ASRSessionPoisoned as exc:
            await self.fail("session_poisoned", "ASR session can no longer be used", 1011)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "ASR inference failed", 1011)
            raise _StreamClosed from exc

        queue_wait, inference_time = self.coordinator.session_timing(self.session_id)
        if queue_wait + inference_time > self.settings.asr_max_connection_lag_seconds:
            await self.fail("realtime_lag_exceeded", "ASR can no longer keep up in real time", 1013)
            raise _StreamClosed
        try:
            await self._send_events(
                self.transcript.apply_model_update(
                    result.text,
                    processed_samples=sample_count,
                )
            )
        except ConfirmedPrefixConflict as exc:
            await self.fail("transcript_conflict", "Model output conflicts with confirmed text", 1011)
            raise _StreamClosed from exc

        if self.silence.add_audio(pcm_bytes, 16000):
            events = self.transcript.commit_pending()
            await self._send_events(events)
            if events:
                await self.coordinator.reset_segment(self.session_id)

    async def reset_segment(self) -> None:
        assert self.session_id is not None
        await self.coordinator.reset_segment(self.session_id)
        self.transcript.reset_segment()
        self.silence.reset()

    async def finish(self) -> None:
        assert self.session_id is not None
        try:
            result = await self.coordinator.finish_stream(self.session_id)
            await self._send_events(self.transcript.finish(result.text))
        except ConfirmedPrefixConflict as exc:
            await self.fail("transcript_conflict", "Model output conflicts with confirmed text", 1011)
            raise _StreamClosed from exc
        except (ASRQueueFull, ASRQueueTimeout) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRInferenceTimeout as exc:
            await self.fail("inference_timeout", "ASR inference timed out", 1011)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "ASR finalization failed", 1011)
            raise _StreamClosed from exc
        self.finished = True
        self.session_id = None
        await self.websocket.close(code=1000)

    async def abort(self) -> None:
        if self.session_id is None:
            return
        session_id, self.session_id = self.session_id, None
        try:
            await self.coordinator.abort_stream(session_id)
        except Exception:
            return

    async def fail(self, code: str, message: str, close_code: int) -> None:
        event = self.transcript.new_event("error")
        await self.websocket.send_json(
            {
                "type": "error",
                "code": code,
                "message": message,
                "sequence": event.sequence,
            }
        )
        await self.websocket.close(code=close_code)

    async def _send_events(self, events) -> None:
        for event in events:
            await self._send_event(event)

    async def _send_event(self, event) -> None:
        payload = {"type": event.type, "sequence": event.sequence}
        if event.type in {"partial", "sentence_final", "final"}:
            payload["text"] = event.text
        await self.websocket.send_json(payload)


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(
    websocket: WebSocket,
    current_settings: Settings = Depends(get_settings),
    current_coordinator: ASRInferenceCoordinator = Depends(get_asr_coordinator),
) -> None:
    await websocket.accept()
    controller = StreamingSessionController(websocket, current_settings, current_coordinator)

    try:
        try:
            async with asyncio.timeout(current_settings.asr_start_timeout_seconds):
                raw_start = await websocket.receive_text()
        except TimeoutError:
            await controller.fail("start_timeout", "Start message was not received in time", 1008)
            return
        except Exception:
            await controller.fail("invalid_start", "Expected a JSON start message", 1003)
            return

        start = await _parse_stream_start(raw_start, controller, current_settings)
        if start is None:
            return

        if current_settings.asr_stream_mode != "stateful":
            committer = SentenceCommitter(commit_on_punctuation=current_settings.asr_commit_on_punctuation)
            silence_detector = SilenceEndpointDetector(
                silence_seconds=current_settings.asr_vad_silence_seconds,
                rms_threshold=current_settings.asr_vad_rms_threshold,
            )
            await _run_chunked_transcribe_stream(
                websocket,
                current_settings,
                start.language,
                start.sample_rate,
                committer,
                silence_detector,
            )
            return

        await controller.start(start.language)
        session_started = time.monotonic()
        while True:
            session_remaining = current_settings.asr_max_session_seconds - (
                time.monotonic() - session_started
            )
            if session_remaining <= 0:
                await controller.fail("session_timeout", "Maximum session duration exceeded", 1008)
                return
            receive_timeout = min(current_settings.asr_idle_timeout_seconds, session_remaining)
            try:
                async with asyncio.timeout(receive_timeout):
                    message = await websocket.receive()
            except TimeoutError:
                if session_remaining <= current_settings.asr_idle_timeout_seconds:
                    await controller.fail("session_timeout", "Maximum session duration exceeded", 1008)
                else:
                    await controller.fail("idle_timeout", "No audio or command was received in time", 1008)
                return

            if message.get("type") == "websocket.disconnect":
                return
            if "bytes" in message and message["bytes"] is not None:
                await controller.add_audio(message["bytes"])
                continue
            if "text" not in message or message["text"] is None:
                await controller.fail("invalid_message", "Expected audio or a JSON command", 1003)
                return
            try:
                payload = json.loads(message["text"])
            except (json.JSONDecodeError, TypeError):
                await controller.fail("invalid_message", "Expected a JSON command", 1003)
                return
            if not isinstance(payload, dict):
                await controller.fail("invalid_message", "Expected a JSON object", 1003)
                return
            if payload.get("type") == "end":
                await controller.finish()
                return
            if payload.get("type") == "segment":
                await controller.reset_segment()
                continue
            await controller.fail("invalid_message", "Unsupported stream command", 1003)
            return
    except _StreamClosed:
        return
    except WebSocketDisconnect:
        return
    finally:
        await controller.abort()


async def _parse_stream_start(
    raw_start: str,
    controller: StreamingSessionController,
    current_settings: Settings,
) -> ASRStreamStart | None:
    try:
        payload = json.loads(raw_start)
    except (json.JSONDecodeError, TypeError):
        await controller.fail("invalid_start", "Expected a JSON start message", 1003)
        return None
    if not isinstance(payload, dict):
        await controller.fail("invalid_start", "Expected a JSON object", 1003)
        return None
    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        await controller.fail("invalid_language", "Language must be a string or null", 1003)
        return None
    sample_rate = payload.get("sample_rate", 16000)
    if isinstance(sample_rate, int) and not isinstance(sample_rate, bool) and sample_rate != 16000:
        await controller.fail("unsupported_sample_rate", "Only 16000 Hz audio is supported", 1003)
        return None
    try:
        start = ASRStreamStart.model_validate(payload)
    except ValidationError:
        await controller.fail("invalid_start", "Invalid start message", 1003)
        return None
    if start.sample_rate != 16000:
        await controller.fail("unsupported_sample_rate", "Only 16000 Hz audio is supported", 1003)
        return None
    if not is_valid_api_key(start.api_key, current_settings):
        await controller.fail("invalid_api_key", "Invalid or missing API key", 1008)
        return None
    return start


async def _run_chunked_transcribe_stream(
    websocket: WebSocket,
    current_settings: Settings,
    language: str | None,
    sample_rate: int,
    committer: SentenceCommitter,
    silence_detector: SilenceEndpointDetector,
) -> None:
    min_chunk_bytes = max(1, int(sample_rate * 2 * current_settings.asr_stream_chunk_seconds))
    buffer = bytearray()

    while True:
        message = await websocket.receive()
        if "bytes" in message and message["bytes"] is not None:
            pcm_bytes = message["bytes"]
            buffer.extend(pcm_bytes)
            if len(buffer) >= min_chunk_bytes:
                segment_text = _transcribe_pcm_chunk(bytes(buffer), sample_rate, language)
                buffer.clear()
                if segment_text:
                    await _send_committed_and_partial(websocket, committer, segment_text)
            if silence_detector.add_audio(pcm_bytes, sample_rate):
                await _send_pending_commit(websocket, committer)
        elif "text" in message and message["text"] is not None:
            payload = json.loads(message["text"])
            if payload.get("type") == "end":
                if buffer:
                    segment_text = _transcribe_pcm_chunk(bytes(buffer), sample_rate, language)
                    if segment_text:
                        await _send_committed_and_partial(websocket, committer, segment_text)
                await websocket.send_json({"type": "final", "text": committer.pending_text})
                await websocket.close(code=1000)
                return
            if payload.get("type") == "segment":
                buffer.clear()
                silence_detector.reset()
        elif message.get("type") == "websocket.disconnect":
            return


async def _run_stateful_transcribe_stream(
    websocket: WebSocket,
    language: str | None,
    sample_rate: int,
    committer: SentenceCommitter,
    stable_committer: StablePunctuationCommitter,
    silence_detector: SilenceEndpointDetector,
) -> None:
    if sample_rate != 16000:
        await websocket.send_json({"type": "error", "message": "Stateful ASR streaming requires sample_rate 16000"})
        await websocket.close(code=1003)
        return

    try:
        session = asr_transcriber.create_streaming_session(language=language)
    except NotImplementedError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1003)
        return

    await websocket.send_json({"type": "ready"})

    while True:
        message = await websocket.receive()
        if "bytes" in message and message["bytes"] is not None:
            pcm_bytes = message["bytes"]
            result = session.add_pcm_s16le(pcm_bytes, sample_rate)
            if stable_committer.enabled:
                await _send_stateful_update(websocket, committer, stable_committer, result.text)
            else:
                await _send_committed_and_partial(websocket, committer, result.text, cumulative=True)
            if silence_detector.add_audio(pcm_bytes, sample_rate):
                await _send_pending_commit(websocket, committer)
                stable_committer.reset()
        elif "text" in message and message["text"] is not None:
            payload = json.loads(message["text"])
            if payload.get("type") == "end":
                result = session.finish()
                if stable_committer.enabled:
                    await _send_stateful_update(websocket, committer, stable_committer, result.text)
                else:
                    await _send_committed_and_partial(websocket, committer, result.text, cumulative=True)
                await websocket.send_json({"type": "final", "text": committer.pending_text})
                await websocket.close(code=1000)
                return
            if payload.get("type") == "segment":
                session.reset_segment()
                silence_detector.reset()
                stable_committer.reset()
        elif message.get("type") == "websocket.disconnect":
            return


async def _send_stateful_update(
    websocket: WebSocket,
    committer: SentenceCommitter,
    stable_committer: StablePunctuationCommitter,
    text: str,
) -> None:
    previous_pending = committer.pending_text
    committer.append_cumulative(text)
    stable_prefix = stable_committer.observe(committer.pending_text, now=time.monotonic())

    if stable_prefix:
        sentence = committer.commit_prefix(stable_prefix)
        if sentence:
            await websocket.send_json({"type": "sentence_final", "text": sentence})

    if committer.pending_text != previous_pending or stable_prefix:
        if committer.pending_text:
            await websocket.send_json({"type": "partial", "text": committer.pending_text})
        elif not stable_prefix and previous_pending:
            await websocket.send_json({"type": "partial", "text": ""})


async def _send_committed_and_partial(
    websocket: WebSocket,
    committer: SentenceCommitter,
    text: str,
    *,
    cumulative: bool = False,
) -> None:
    previous_pending = committer.pending_text
    append = committer.append_cumulative if cumulative else committer.append
    for sentence in append(text):
        await websocket.send_json({"type": "sentence_final", "text": sentence})
    if committer.pending_text != previous_pending and (committer.pending_text or cumulative):
        await websocket.send_json({"type": "partial", "text": committer.pending_text})


async def _send_pending_commit(websocket: WebSocket, committer: SentenceCommitter) -> None:
    sentence = committer.commit_pending()
    if sentence:
        await websocket.send_json({"type": "sentence_final", "text": sentence})


def _transcribe_pcm_chunk(pcm_bytes: bytes, sample_rate: int, language: str | None) -> str:
    temp_path = write_pcm_s16le_wav(pcm_bytes, sample_rate)
    try:
        result = asr_transcriber.transcribe(temp_path, language=language)
        return result.text
    finally:
        remove_file(temp_path)
