from __future__ import annotations

import asyncio
import json
import logging
import queue
import struct
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Event

from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect

from app.auth import is_valid_api_key, require_api_key
from app.config import Settings, get_settings
from app.schemas import TTSCapacityResponse, TTSHealthResponse, TTSInfoResponse, TTSRequest
from app.tts import TTSSynthesizer, create_tts_synthesizer

settings = get_settings()
tts_synthesizer = create_tts_synthesizer(settings)
tts_synthesizers = {
    settings.tts_model_name: tts_synthesizer,
}
tts_stream_slots = BoundedSemaphore(settings.tts_max_active_streams)

_STREAM_END = object()
stream_logger = logging.getLogger("uvicorn.error.tts.stream")


@dataclass(frozen=True)
class _StreamFailure:
    error: Exception


@asynccontextmanager
async def lifespan(_app: FastAPI):
    starter = getattr(tts_synthesizer, "start", None)
    if callable(starter):
        await asyncio.to_thread(starter)
    warmer = getattr(tts_synthesizer, "warmup", None)
    if callable(warmer):
        await asyncio.to_thread(warmer)
    try:
        yield
    finally:
        closer = getattr(tts_synthesizer, "close", None)
        if callable(closer):
            await asyncio.to_thread(closer)


app = FastAPI(
    title="TTS REST API",
    version="0.3.0",
    description="REST and MiniMax-style WebSocket API for streaming text-to-speech.",
    lifespan=lifespan,
)


def get_tts_synthesizer() -> TTSSynthesizer:
    return tts_synthesizer


def get_tts_synthesizers() -> dict[str, TTSSynthesizer]:
    return tts_synthesizers


@app.get("/health", response_model=TTSHealthResponse)
def health(current_settings: Settings = Depends(get_settings)) -> TTSHealthResponse:
    return TTSHealthResponse(
        status="ok",
        model=current_settings.tts_model_name,
        backend=current_settings.tts_backend,
        sample_rate=current_settings.tts_sample_rate,
    )


@app.get("/v1/tts/info", response_model=TTSInfoResponse)
def tts_info(current_settings: Settings = Depends(get_settings)) -> TTSInfoResponse:
    return TTSInfoResponse(
        websocket_url="/v1/tts/stream",
        http_endpoint="/v1/tts",
        authentication={
            "header": "Authorization",
            "scheme": "Bearer",
            "alternative_header": "X-API-Key",
        },
        task_start={
            "event": "task_start",
            "model": current_settings.tts_model_name,
            "voice_setting": {"voice_id": current_settings.tts_default_voice},
            "audio_setting": {
                "sample_rate": current_settings.tts_sample_rate,
                "format": "pcm",
                "channel": 1,
            },
            "stream_options": {"audio_transport": "hex"},
        },
        task_continue={"event": "task_continue", "text": "..."},
        task_finish={"event": "task_finish"},
        server_events=[
            "connected_success",
            "task_started",
            "task_continued",
            "task_finished",
            "task_failed",
        ],
        audio_transports=["hex", "binary"],
    )


@app.get(
    "/v1/tts/capacity",
    response_model=TTSCapacityResponse,
    dependencies=[Depends(require_api_key)],
)
def tts_capacity(
    current_settings: Settings = Depends(get_settings),
    current_synthesizer: TTSSynthesizer = Depends(get_tts_synthesizer),
) -> TTSCapacityResponse:
    snapshot = _capacity_snapshot(current_synthesizer, current_settings)
    return TTSCapacityResponse(
        status="ok",
        model=current_settings.tts_model_name,
        backend=current_settings.tts_backend,
        sample_rate=current_settings.tts_sample_rate,
        **snapshot,
    )


@app.post("/v1/tts", dependencies=[Depends(require_api_key)])
def synthesize_tts(
    request: TTSRequest,
    current_settings: Settings = Depends(get_settings),
    current_synthesizer: TTSSynthesizer = Depends(get_tts_synthesizer),
) -> Response:
    _validate_text_length(request.text, current_settings)
    voice = request.voice or current_settings.tts_default_voice

    try:
        audio = current_synthesizer.synthesize(request.text, voice=voice)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/wav")


def _capacity_snapshot(
    synthesizer: TTSSynthesizer,
    current_settings: Settings,
) -> dict[str, object]:
    snapshot = getattr(synthesizer, "capacity_snapshot", None)
    if callable(snapshot):
        return snapshot()
    supports_realtime_streaming = current_settings.tts_backend in {
        "mock",
        "cosyvoice",
        "triton",
        "vllm_omni",
        "voxserve",
    }
    return {
        "ready": getattr(synthesizer, "ready", None),
        "supports_realtime_streaming": supports_realtime_streaming,
        "supports_microbatch": False,
        "queue_depth": None,
        "queue_size": None,
        "batch_size": None,
        "batch_wait_ms": None,
        "dispatched_batches": None,
        "dispatched_requests": None,
        "last_batch_size": None,
    }


@app.websocket("/v1/tts/stream")
async def synthesize_tts_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    current_settings = get_settings()
    session_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    if not is_valid_api_key(_websocket_api_key(websocket), current_settings):
        await _fail_websocket(
            websocket,
            session_id,
            trace_id,
            1001,
            "Invalid or missing API key",
            close_code=1008,
        )
        return

    await websocket.send_json(
        _event_payload("connected_success", session_id, trace_id)
    )

    try:
        start = await _receive_json(websocket)
        if start.get("event") != "task_start":
            raise ValueError("Expected task_start event")
        model, voice, transport = _validate_task_start(start, current_settings)
        current_synthesizer = tts_synthesizers.get(model)
        if current_synthesizer is None:
            raise ValueError(f"unknown TTS model: {model}")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        await _fail_websocket(
            websocket,
            session_id,
            trace_id,
            1003,
            str(exc),
            close_code=1003,
        )
        return
    except WebSocketDisconnect:
        return

    if not tts_stream_slots.acquire(blocking=False):
        await _fail_websocket(
            websocket,
            session_id,
            trace_id,
            1013,
            "TTS is at capacity",
            close_code=1013,
        )
        return

    chunk_sequence = 0
    sample_offset = 0
    text_characters = 0
    synthesis_ms = 0.0
    encode_ms = 0.0
    saw_text = False
    last_chunk_ready_at: float | None = None
    last_chunk_sent_at: float | None = None
    max_chunk_ready_gap_ms = 0.0
    max_chunk_sent_gap_ms = 0.0

    try:
        await websocket.send_json(_event_payload("task_started", session_id, trace_id))
        while True:
            payload = await _receive_json(websocket)
            event = payload.get("event")

            if event == "task_finish":
                if not saw_text:
                    raise ValueError("task_finish requires at least one task_continue event")
                await websocket.send_json(
                    {
                        **_event_payload("task_finished", session_id, trace_id),
                        "extra_info": {
                            "chunks": chunk_sequence,
                            "total_samples": sample_offset,
                            "audio_sample_rate": current_settings.tts_sample_rate,
                            "audio_channel": 1,
                            "queue_ms": None,
                            "synthesis_ms": round(synthesis_ms, 3),
                            "encode_ms": round(encode_ms, 3),
                        },
                    }
                )
                stream_logger.info(
                    "completed trace_id=%s chunks=%d max_chunk_ready_gap_ms=%.3f "
                    "max_chunk_sent_gap_ms=%.3f",
                    trace_id,
                    chunk_sequence,
                    max_chunk_ready_gap_ms,
                    max_chunk_sent_gap_ms,
                )
                await websocket.close(code=1000)
                return

            if event != "task_continue":
                raise ValueError("Expected task_continue or task_finish event")

            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("task_continue.text must be a string")
            _validate_text_length(text, current_settings)
            text_characters += len(text)
            if text_characters > current_settings.tts_max_text_chars:
                raise ValueError(
                    f"text exceeds maximum length of {current_settings.tts_max_text_chars} characters"
                )
            saw_text = True

            synthesis_started = time.perf_counter()
            encode_before = encode_ms
            returned_audio = False
            async for pcm in _stream_pcm_in_thread(current_synthesizer, text, voice):
                returned_audio = True
                chunk_ready_at = time.perf_counter()
                ready_gap_ms = 0.0
                if last_chunk_ready_at is not None:
                    ready_gap_ms = (chunk_ready_at - last_chunk_ready_at) * 1000
                    max_chunk_ready_gap_ms = max(max_chunk_ready_gap_ms, ready_gap_ms)
                encode_started = time.perf_counter()
                if transport == "binary":
                    header = struct.pack("<4sIQ", b"TTS1", chunk_sequence, sample_offset)
                    await websocket.send_bytes(header + pcm)
                else:
                    await websocket.send_json(
                        {
                            **_event_payload("task_continued", session_id, trace_id),
                            "data": {"audio": pcm.hex()},
                            "is_final": False,
                            "extra_info": {
                                "audio_format": "pcm",
                                "audio_sample_rate": current_settings.tts_sample_rate,
                                "audio_channel": 1,
                                "chunk_sequence": chunk_sequence,
                                "sample_offset": sample_offset,
                            },
                        }
                    )
                encode_ms += (time.perf_counter() - encode_started) * 1000
                chunk_sent_at = time.perf_counter()
                sent_gap_ms = 0.0
                if last_chunk_sent_at is not None:
                    sent_gap_ms = (chunk_sent_at - last_chunk_sent_at) * 1000
                    max_chunk_sent_gap_ms = max(max_chunk_sent_gap_ms, sent_gap_ms)
                gap_trace_ms = current_settings.tts_stream_gap_trace_ms
                if gap_trace_ms and max(ready_gap_ms, sent_gap_ms) >= gap_trace_ms:
                    stream_logger.info(
                        "TTS gap trace: trace_id=%s chunk_sequence=%d ready_gap_ms=%.3f "
                        "sent_gap_ms=%.3f pcm_bytes=%d",
                        trace_id,
                        chunk_sequence,
                        ready_gap_ms,
                        sent_gap_ms,
                        len(pcm),
                    )
                last_chunk_ready_at = chunk_ready_at
                last_chunk_sent_at = chunk_sent_at
                chunk_sequence += 1
                sample_offset += len(pcm) // 2
            if not returned_audio:
                raise RuntimeError("TTS did not return audio")
            elapsed_ms = (time.perf_counter() - synthesis_started) * 1000
            synthesis_ms += elapsed_ms - (encode_ms - encode_before)
    except WebSocketDisconnect:
        return
    except HTTPException as exc:
        await _fail_websocket(websocket, session_id, trace_id, 1004, str(exc.detail))
    except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        await _fail_websocket(websocket, session_id, trace_id, 1004, str(exc))
    finally:
        tts_stream_slots.release()


def _websocket_api_key(websocket: WebSocket) -> str | None:
    authorization = websocket.headers.get("authorization")
    if authorization:
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            return token.strip()
    return websocket.headers.get("x-api-key")


def _validate_task_start(payload: dict, settings: Settings) -> tuple[str, str, str]:
    model = payload.get("model")
    if model != settings.tts_model_name:
        raise ValueError(
            f"unknown TTS model: {model}; configured model is {settings.tts_model_name}"
        )

    voice_setting = payload.get("voice_setting")
    if not isinstance(voice_setting, dict):
        raise ValueError("voice_setting must be an object")
    voice = voice_setting.get("voice_id") or settings.tts_default_voice
    if not isinstance(voice, str):
        raise ValueError("voice_setting.voice_id must be a string")
    for field, default in (("speed", 1.0), ("vol", 1.0), ("pitch", 0)):
        if voice_setting.get(field, default) != default:
            raise ValueError(f"voice_setting.{field} is not supported by this backend")

    audio_setting = payload.get("audio_setting")
    if not isinstance(audio_setting, dict):
        raise ValueError("audio_setting must be an object")
    if audio_setting.get("sample_rate") != settings.tts_sample_rate:
        raise ValueError(f"Only {settings.tts_sample_rate} Hz output is supported")
    if audio_setting.get("format") != "pcm":
        raise ValueError("Only pcm stream format is supported")
    if audio_setting.get("channel") != 1:
        raise ValueError("Only mono output is supported")

    stream_options = payload.get("stream_options", {})
    if not isinstance(stream_options, dict):
        raise ValueError("stream_options must be an object")
    transport = stream_options.get("audio_transport", "hex")
    if transport not in {"hex", "binary"}:
        raise ValueError("audio_transport must be hex or binary")
    return model, voice, transport


async def _receive_json(websocket: WebSocket) -> dict:
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if text is None:
        raise ValueError("Expected a JSON text message")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


async def _fail_websocket(
    websocket: WebSocket,
    session_id: str,
    trace_id: str,
    status_code: int,
    message: str,
    close_code: int = 1011,
) -> None:
    try:
        await websocket.send_json(
            _event_payload("task_failed", session_id, trace_id, status_code, message)
        )
        await websocket.close(code=close_code)
    except (RuntimeError, WebSocketDisconnect):
        return


def _event_payload(
    event: str,
    session_id: str,
    trace_id: str,
    status_code: int = 0,
    status_msg: str = "success",
) -> dict:
    return {
        "event": event,
        "session_id": session_id,
        "trace_id": trace_id,
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


async def _stream_pcm_in_thread(
    synthesizer: TTSSynthesizer,
    text: str,
    voice: str,
) -> AsyncIterator[bytes]:
    items: queue.Queue[object] = queue.Queue(maxsize=2)
    stopped = Event()

    def put(item: object) -> bool:
        while not stopped.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for pcm in synthesizer.stream_pcm(text, voice=voice):
                if not pcm:
                    continue
                if len(pcm) % 2:
                    raise RuntimeError("TTS produced an odd-length pcm_s16le chunk")
                if not put(pcm):
                    return
        except Exception as exc:
            if not put(_StreamFailure(exc)):
                return
        put(_STREAM_END)

    producer = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await asyncio.to_thread(items.get)
            if item is _STREAM_END:
                return
            if isinstance(item, _StreamFailure):
                raise RuntimeError(str(item.error)) from item.error
            assert isinstance(item, bytes)
            yield item
    finally:
        stopped.set()
        await producer


def _validate_text_length(text: str, settings: Settings) -> None:
    if not text.strip():
        raise HTTPException(status_code=422, detail="text cannot be blank")

    if len(text) > settings.tts_max_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds maximum length of {settings.tts_max_text_chars} characters",
        )
