from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
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

from app.asr import create_asr_transcriber
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
    ASRSessionBusy,
    ASRSessionPoisoned,
)
from app.asr_streaming import (
    ConfirmedPrefixConflict,
    SilenceEndpointDetector,
    StreamingTranscriptState,
)
from app.asr_vad import VADDecision, create_vad_endpoint_detector
from app.schemas import (
    ASRHealthResponse,
    ASRReadyResponse,
    TranscribeResponse,
    TranscribeStreamInfoResponse,
)

logger = logging.getLogger(__name__)

settings = get_settings()
asr_coordinator = ASRInferenceCoordinator(settings, lambda: create_asr_transcriber(settings))


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


def get_asr_coordinator() -> ASRInferenceCoordinator:
    return asr_coordinator


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
    stateful = current_settings.asr_stream_mode == "stateful"
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
            "commit_on_punctuation": (
                current_settings.asr_commit_on_punctuation and not stateful
            ),
            "endpointing": {
                "backend": (
                    "silero_onnx_cpu"
                    if stateful and current_settings.asr_backend == "qwen_vllm"
                    else "legacy_chunked_or_mock"
                ),
                "immutable_commit_source": (
                    "vad_endpoint_explicit_segment_or_forced_boundary"
                    if stateful
                    else "chunked_legacy_policy"
                ),
                "onset_threshold": current_settings.asr_vad_onset_threshold,
                "offset_threshold": current_settings.asr_vad_offset_threshold,
                "min_speech_ms": current_settings.asr_vad_min_speech_ms,
                "min_silence_ms": current_settings.asr_vad_min_silence_ms,
                "hangover_ms": current_settings.asr_vad_hangover_ms,
                "pre_roll_ms": current_settings.asr_vad_pre_roll_ms,
                "model_version": current_settings.asr_vad_model_version,
                "model_sha256": current_settings.asr_vad_model_sha256,
                "normal_utterance_seconds": current_settings.asr_max_utterance_seconds,
                "invariant_watchdog_seconds": current_settings.asr_state_watchdog_seconds,
            },
            "stateful": {
                "chunk_seconds": current_settings.asr_stream_chunk_seconds,
                "unfixed_chunk_num": current_settings.asr_stream_unfixed_chunk_num,
                "unfixed_token_num": current_settings.asr_stream_unfixed_token_num,
                "rollover_seconds": (
                    current_settings.asr_max_utterance_seconds
                    if stateful
                    else current_settings.asr_stream_rollover_seconds
                ),
                "vllm_gpu_memory_utilization": current_settings.asr_vllm_gpu_memory_utilization,
                "vllm_max_new_tokens": current_settings.asr_vllm_max_new_tokens,
                "stable_commit_enabled": (
                    current_settings.asr_stable_commit_enabled and not stateful
                ),
                "stable_commit_seconds": current_settings.asr_stable_commit_seconds,
                "stable_commit_min_chars": current_settings.asr_stable_commit_min_chars,
                "stable_commit_min_updates": current_settings.asr_stable_commit_min_updates,
                "legacy_commit_options_requested": {
                    "commit_on_punctuation": current_settings.asr_commit_on_punctuation,
                    "stable_commit_enabled": current_settings.asr_stable_commit_enabled,
                },
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
        vad_detector=None,
    ) -> None:
        self.websocket = websocket
        self.settings = current_settings
        self.coordinator = coordinator
        self.session_id: str | None = None
        self.finished = False
        self.accepted_samples = 0
        self.processing_debt_seconds = 0.0
        self.oldest_undecoded_age_seconds = 0.0
        self._model_work_seconds = 0.0
        self._undecoded_batches: deque[list[float]] = deque()
        self._lag_error_sent = False
        self.stateful = current_settings.asr_stream_mode == "stateful"
        self.session_deadline = time.monotonic() + current_settings.asr_max_session_seconds
        self.transcript = StreamingTranscriptState(
            sample_rate=16000,
            stable_commit_enabled=current_settings.asr_stable_commit_enabled,
            stable_commit_seconds=current_settings.asr_stable_commit_seconds,
            stable_commit_min_chars=current_settings.asr_stable_commit_min_chars,
            stable_commit_min_updates=current_settings.asr_stable_commit_min_updates,
            immediate_commit_on_punctuation=(
                current_settings.asr_commit_on_punctuation
                and current_settings.asr_stream_mode == "chunked"
            ),
            segment_local_snapshots=self.stateful,
        )
        self.silence = SilenceEndpointDetector(
            silence_seconds=current_settings.asr_vad_silence_seconds,
            rms_threshold=current_settings.asr_vad_rms_threshold,
        )
        self.vad = vad_detector
        if (
            self.vad is None
            and self.stateful
            and current_settings.asr_backend == "qwen_vllm"
        ):
            self.vad = create_vad_endpoint_detector(current_settings)

    async def start(self, language: str | None) -> None:
        await self._start_with(self.coordinator.create_stream, language)

    async def start_chunked(self, language: str | None) -> None:
        await self._start_with(self.coordinator.create_chunked_stream, language)

    async def _start_with(self, create_session, language: str | None) -> None:
        try:
            self.session_id = await self._await_model_operation(
                lambda: create_session(language)
            )
        except _StreamClosed:
            raise
        except ValueError as exc:
            await self.fail("invalid_language", "Unsupported language", 1003)
            raise _StreamClosed from exc
        except (
            ASRQueueFull,
            ASRQueueTimeout,
            ASRSessionLimit,
            ASRSessionBusy,
            ASRBatchConflict,
        ) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRNotReady as exc:
            await self.fail("not_ready", "ASR is not ready", 1013)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "Unable to create ASR session", 1011)
            raise _StreamClosed from exc
        await self._ensure_session_active()
        logger.info(
            "asr_stream_started session_id=%s active_streams=%d",
            self.session_id,
            self.coordinator.snapshot().active_streams,
        )
        await self._send_event(self.transcript.new_event("ready"))

    async def add_audio(self, pcm_bytes: bytes) -> None:
        await self.validate_audio_frame(pcm_bytes)
        if self.vad is not None:
            decision = self.vad.add_audio(pcm_bytes)
            while True:
                await self._handle_vad_decision(decision)
                if not decision.endpoint:
                    return
                decision = self.vad.endpoint_finalized()
        result = await self._add_model_audio(pcm_bytes)
        silence_endpoint = self.silence.add_audio(pcm_bytes, 16000)
        if result.segment_finished:
            await self._commit_finished_segment(reset_endpoint_detector=False)
        elif silence_endpoint:
            await self.reset_segment()

    async def _handle_vad_decision(self, decision: VADDecision) -> None:
        if decision.discarded_samples:
            logger.info(
                "asr_vad_discarded session_id=%s samples=%d duration_seconds=%.3f",
                self.session_id or "unassigned",
                decision.discarded_samples,
                decision.discarded_samples / 16000,
            )
        result = None
        if decision.audio_to_model:
            result = await self._add_vad_audio(decision.audio_to_model)
        if decision.endpoint and not (result and result.segment_finished):
            await self.reset_segment(reset_endpoint_detector=False)

    async def explicit_segment(self) -> None:
        already_finalized = False
        if self.vad is not None:
            pending = self.vad.finish_input()
            if pending.audio_to_model:
                result = await self._add_vad_audio(pending.audio_to_model)
                if result and result.segment_finished:
                    already_finalized = True
        if not already_finalized:
            await self.reset_segment(reset_endpoint_detector=False)
        if self.vad is not None:
            self.vad.reset()

    async def _add_model_audio(self, pcm_bytes: bytes):
        assert self.session_id is not None
        try:
            result = await self._await_model_operation(
                lambda: self.coordinator.add_audio(self.session_id, pcm_bytes, 16000)
            )
        except _StreamClosed:
            raise
        except (
            ASRQueueFull,
            ASRQueueTimeout,
            ASRSessionLimit,
            ASRSessionBusy,
            ASRBatchConflict,
        ) as exc:
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

        await self._ensure_session_active()
        if self._observe_stream_progress(
            result, submitted_samples=len(pcm_bytes) // 2
        ):
            await self.fail("realtime_lag_exceeded", "ASR can no longer keep up in real time", 1013)
            raise _StreamClosed
        try:
            if result.model_updated:
                await self._apply_stream_result(result)
        except ConfirmedPrefixConflict as exc:
            await self.fail("transcript_conflict", "Model output conflicts with confirmed text", 1011)
            raise _StreamClosed from exc

        return result

    async def _add_vad_audio(self, pcm_bytes: bytes):
        result = None
        frame_bytes = self.settings.asr_max_frame_bytes
        for offset in range(0, len(pcm_bytes), frame_bytes):
            result = await self._add_model_audio(
                pcm_bytes[offset : offset + frame_bytes]
            )
            if result.segment_finished:
                await self._commit_finished_segment(
                    reset_endpoint_detector=False
                )
        return result

    async def validate_audio_frame(self, pcm_bytes: bytes) -> int:
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
        return sample_count

    async def transcribe_chunk(self, pcm_bytes: bytes, language: str | None) -> str:
        temp_path = await asyncio.to_thread(write_pcm_s16le_wav, pcm_bytes, 16000)
        started = time.monotonic()
        try:
            assert self.session_id is not None
            result = await self._await_model_operation(
                lambda: self.coordinator.transcribe_stream_chunk(
                    self.session_id,
                    temp_path,
                    language,
                    (len(pcm_bytes) // 2) / 16000,
                )
            )
        except _StreamClosed:
            raise
        except (ASRQueueFull, ASRQueueTimeout, ASRSessionBusy, ASRBatchConflict) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRInferenceTimeout as exc:
            await self.fail("inference_timeout", "ASR inference timed out", 1011)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "ASR inference failed", 1011)
            raise _StreamClosed from exc
        finally:
            remove_file(temp_path)
        await self._ensure_session_active()
        elapsed = time.monotonic() - started
        if self._lag_exceeded(elapsed, (len(pcm_bytes) // 2) / 16000):
            await self.fail("realtime_lag_exceeded", "ASR can no longer keep up in real time", 1013)
            raise _StreamClosed
        return result.text

    def _lag_exceeded(self, processing_seconds: float, audio_seconds: float) -> bool:
        self.processing_debt_seconds = max(
            0.0,
            self.processing_debt_seconds + processing_seconds - audio_seconds,
        )
        return self.processing_debt_seconds > self.settings.asr_max_connection_lag_seconds

    def _observe_stream_progress(self, result, *, submitted_samples: int) -> bool:
        now = time.monotonic()
        if submitted_samples:
            self._undecoded_batches.append(
                [float(submitted_samples), now, self._model_work_seconds]
            )
        processing_seconds = max(0.0, result.queue_wait_seconds) + max(
            0.0, result.inference_seconds
        )
        self._model_work_seconds += processing_seconds
        decoded_samples = result.decoded_samples_delta
        decoded_seconds = (
            max(0, decoded_samples) / 16000
            if decoded_samples is not None
            else 0.0
        )
        self.processing_debt_seconds = max(
            0.0,
            self.processing_debt_seconds + processing_seconds - decoded_seconds,
        )

        remaining_decoded = max(0, decoded_samples or 0)
        while remaining_decoded and self._undecoded_batches:
            batch = self._undecoded_batches[0]
            consumed = min(remaining_decoded, int(batch[0]))
            batch[0] -= consumed
            remaining_decoded -= consumed
            if batch[0] <= 0:
                self._undecoded_batches.popleft()

        if self._undecoded_batches:
            oldest = self._undecoded_batches[0]
            self.oldest_undecoded_age_seconds = max(
                0.0,
                now - oldest[1],
                self._model_work_seconds - oldest[2],
            )
        else:
            self.oldest_undecoded_age_seconds = 0.0

        logger.info(
            "asr_stream_progress session_id=%s decoded_audio_seconds=%.3f queue_wait_ms=%.3f inference_ms=%.3f lag_debt_seconds=%.3f oldest_undecoded_age_seconds=%.3f",
            self.session_id or "unassigned",
            decoded_seconds,
            max(0.0, result.queue_wait_seconds) * 1000,
            max(0.0, result.inference_seconds) * 1000,
            self.processing_debt_seconds,
            self.oldest_undecoded_age_seconds,
        )
        exceeded = (
            self.processing_debt_seconds
            > self.settings.asr_max_connection_lag_seconds
            and self.oldest_undecoded_age_seconds
            > self.settings.asr_max_undecoded_age_seconds
        )
        if not exceeded or self._lag_error_sent:
            return False
        self._lag_error_sent = True
        return True

    async def _enforce_stream_progress(
        self, result, *, submitted_samples: int = 0
    ) -> None:
        if not self._observe_stream_progress(
            result, submitted_samples=submitted_samples
        ):
            return
        await self.fail(
            "realtime_lag_exceeded",
            "ASR can no longer keep up in real time",
            1013,
        )
        raise _StreamClosed

    async def reset_segment(self, *, reset_endpoint_detector: bool = True) -> None:
        assert self.session_id is not None
        try:
            result = await self._await_model_operation(
                lambda: self.coordinator.finish_segment(self.session_id)
            )
        except _StreamClosed:
            raise
        except (ASRQueueFull, ASRQueueTimeout, ASRSessionBusy, ASRBatchConflict) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRInferenceTimeout as exc:
            await self.fail("inference_timeout", "ASR inference timed out", 1011)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "ASR segment reset failed", 1011)
            raise _StreamClosed from exc
        await self._ensure_session_active()
        await self._enforce_stream_progress(result)
        try:
            await self._apply_stream_result(result)
        except ConfirmedPrefixConflict as exc:
            await self.fail("transcript_conflict", "Model output conflicts with confirmed text", 1011)
            raise _StreamClosed from exc
        await self._commit_finished_segment(
            reset_endpoint_detector=reset_endpoint_detector
        )

    async def _commit_finished_segment(
        self, *, reset_endpoint_detector: bool = True
    ) -> None:
        await self._send_events(self.transcript.commit_pending())
        self.transcript.reset_segment()
        self.silence.reset()
        if reset_endpoint_detector and self.vad is not None:
            self.vad.reset()

    async def finish(self) -> None:
        assert self.session_id is not None
        try:
            if self.vad is not None:
                pending = self.vad.finish_input()
                if pending.audio_to_model:
                    await self._add_vad_audio(pending.audio_to_model)
            result = await self._await_model_operation(
                lambda: self.coordinator.finish_stream(self.session_id)
            )
            await self._ensure_session_active()
            await self._enforce_stream_progress(result)
            await self._apply_stream_result(result)
            await self._send_event(self.transcript.final_event())
        except _StreamClosed:
            raise
        except ConfirmedPrefixConflict as exc:
            await self.fail("transcript_conflict", "Model output conflicts with confirmed text", 1011)
            raise _StreamClosed from exc
        except (ASRQueueFull, ASRQueueTimeout, ASRSessionBusy) as exc:
            await self.fail("server_busy", "ASR is at capacity", 1013)
            raise _StreamClosed from exc
        except ASRInferenceTimeout as exc:
            await self.fail("inference_timeout", "ASR inference timed out", 1011)
            raise _StreamClosed from exc
        except Exception as exc:
            await self.fail("inference_error", "ASR finalization failed", 1011)
            raise _StreamClosed from exc
        self.finished = True
        logger.info("asr_stream_ended session_id=%s reason=finished", self.session_id)
        self.session_id = None
        await self.websocket.close(code=1000)

    async def _apply_stream_result(self, result) -> None:
        if self.stateful:
            events = self.transcript.apply_segment_snapshot(
                result.segment_id,
                result.segment_text,
                decoded_samples_delta=result.decoded_samples_delta or 0,
            )
        else:
            events = self.transcript.apply_model_update(
                result.text,
                processed_samples=result.decoded_samples_delta or 0,
            )
        await self._send_events(events)

    @property
    def remaining_session_seconds(self) -> float:
        return max(0.0, self.session_deadline - time.monotonic())

    async def _await_model_operation(self, operation):
        remaining = self.remaining_session_seconds
        if remaining <= 0:
            await self._expire_session()
        try:
            async with asyncio.timeout(remaining):
                return await operation()
        except TimeoutError:
            await self._expire_session()

    async def _ensure_session_active(self) -> None:
        if self.remaining_session_seconds <= 0:
            await self._expire_session()

    async def _expire_session(self) -> None:
        session_id, self.session_id = self.session_id, None
        if session_id is not None:
            try:
                await self.coordinator.abort_stream(session_id)
            except Exception:
                pass
        await self.fail("session_timeout", "Maximum session duration exceeded", 1008)
        raise _StreamClosed

    async def abort(self) -> None:
        if self.session_id is None:
            return
        session_id, self.session_id = self.session_id, None
        try:
            await self.coordinator.abort_stream(session_id)
        except Exception:
            return
        logger.info("asr_stream_ended session_id=%s reason=aborted", session_id)

    async def fail(self, code: str, message: str, close_code: int) -> None:
        logger.warning(
            "asr_stream_error session_id=%s code=%s close_code=%d",
            self.session_id or "unassigned",
            code,
            close_code,
        )
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
            await self._ensure_session_active()
            await self._send_event(event)

    async def _send_event(self, event) -> None:
        payload = {"type": event.type, "sequence": event.sequence}
        if event.type in {"partial", "sentence_final", "final"}:
            payload["text"] = event.text
        if event.type == "error":
            await self.websocket.send_json(payload)
            return
        remaining = self.remaining_session_seconds
        if remaining <= 0:
            await self._expire_session()
        try:
            async with asyncio.timeout(remaining):
                await self.websocket.send_json(payload)
        except TimeoutError:
            await self._expire_session()


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
            silence_detector = SilenceEndpointDetector(
                silence_seconds=current_settings.asr_vad_silence_seconds,
                rms_threshold=current_settings.asr_vad_rms_threshold,
            )
            await controller.start_chunked(start.language)
            await _run_chunked_transcribe_stream(
                controller,
                current_settings,
                start.language,
                start.sample_rate,
                silence_detector,
            )
            return

        await controller.start(start.language)
        while True:
            session_remaining = controller.remaining_session_seconds
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
                await controller.explicit_segment()
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
    controller: StreamingSessionController,
    current_settings: Settings,
    language: str | None,
    sample_rate: int,
    silence_detector: SilenceEndpointDetector,
) -> None:
    websocket = controller.websocket
    min_chunk_bytes = max(1, int(sample_rate * 2 * current_settings.asr_stream_chunk_seconds))
    buffer = bytearray()
    while True:
        session_remaining = controller.remaining_session_seconds
        if session_remaining <= 0:
            await controller.fail("session_timeout", "Maximum session duration exceeded", 1008)
            raise _StreamClosed
        receive_timeout = min(current_settings.asr_idle_timeout_seconds, session_remaining)
        try:
            async with asyncio.timeout(receive_timeout):
                message = await websocket.receive()
        except TimeoutError:
            if session_remaining <= current_settings.asr_idle_timeout_seconds:
                await controller.fail("session_timeout", "Maximum session duration exceeded", 1008)
            else:
                await controller.fail("idle_timeout", "No audio or command was received in time", 1008)
            raise _StreamClosed
        if message.get("type") == "websocket.disconnect":
            return
        if "bytes" in message and message["bytes"] is not None:
            pcm_bytes = message["bytes"]
            await controller.validate_audio_frame(pcm_bytes)
            buffer.extend(pcm_bytes)
            if len(buffer) >= min_chunk_bytes:
                segment_text = await controller.transcribe_chunk(bytes(buffer), language)
                buffer.clear()
                if segment_text:
                    await controller._send_events(
                        controller.transcript.append_independent_segment(segment_text)
                    )
            if silence_detector.add_audio(pcm_bytes, sample_rate):
                await controller._send_events(controller.transcript.commit_pending())
        elif "text" in message and message["text"] is not None:
            try:
                payload = json.loads(message["text"])
            except json.JSONDecodeError:
                await controller.fail("invalid_message", "Expected a JSON command", 1003)
                raise _StreamClosed
            if not isinstance(payload, dict):
                await controller.fail("invalid_message", "Expected a JSON object", 1003)
                raise _StreamClosed
            if payload.get("type") == "end":
                if buffer:
                    segment_text = await controller.transcribe_chunk(bytes(buffer), language)
                    if segment_text:
                        await controller._send_events(
                            controller.transcript.append_independent_segment(segment_text)
                        )
                await controller._ensure_session_active()
                await controller._send_event(
                    controller.transcript.new_event("final", controller.transcript.partial_text)
                )
                await controller.abort()
                await websocket.close(code=1000)
                return
            if payload.get("type") == "segment":
                buffer.clear()
                silence_detector.reset()
                continue
            await controller.fail("invalid_message", "Unsupported stream command", 1003)
            raise _StreamClosed
        elif message.get("type") == "websocket.disconnect":
            return
