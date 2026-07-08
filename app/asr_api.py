from __future__ import annotations

import json

from fastapi import Depends, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from app.asr import ASRTranscriber, create_asr_transcriber
from app.audio import remove_file, save_upload_to_tempfile, write_pcm_s16le_wav
from app.auth import is_valid_api_key, require_api_key
from app.config import Settings, get_settings
from app.schemas import ASRHealthResponse, TranscribeResponse, TranscribeStreamInfoResponse

settings = get_settings()
asr_transcriber = create_asr_transcriber(settings)

app = FastAPI(
    title="Qwen ASR REST API",
    version="0.1.0",
    description="REST and WebSocket API template for Qwen3-ASR-1.7B deployment.",
)


def get_asr_transcriber() -> ASRTranscriber:
    return asr_transcriber


@app.get("/health", response_model=ASRHealthResponse)
def health(current_settings: Settings = Depends(get_settings)) -> ASRHealthResponse:
    return ASRHealthResponse(
        status="ok",
        model=current_settings.asr_model_name,
        backend=current_settings.asr_backend,
    )


@app.post(
    "/v1/transcribe",
    response_model=TranscribeResponse,
    dependencies=[Depends(require_api_key)],
)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    current_settings: Settings = Depends(get_settings),
    current_transcriber: ASRTranscriber = Depends(get_asr_transcriber),
) -> TranscribeResponse:
    temp_path = await save_upload_to_tempfile(
        file,
        max_bytes=current_settings.asr_max_upload_mb * 1024 * 1024,
    )
    try:
        result = current_transcriber.transcribe(temp_path, language=language)
    finally:
        remove_file(temp_path)

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        model=current_settings.asr_model_name,
    )


@app.get("/v1/transcribe/stream-info", response_model=TranscribeStreamInfoResponse)
def transcribe_stream_info() -> TranscribeStreamInfoResponse:
    return TranscribeStreamInfoResponse(
        websocket_url="/v1/transcribe/stream",
        audio_format={
            "format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "recommended_chunk_ms": "100-500",
        },
        start_message={
            "type": "start",
            "api_key": "<your-api-key>",
            "language": "zh",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        },
        end_message={"type": "end"},
        server_messages=[
            {"type": "ready"},
            {"type": "partial", "text": "..."},
            {"type": "final", "text": "..."},
            {"type": "error", "message": "..."},
        ],
    )


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    current_settings = get_settings()

    try:
        start_message = await websocket.receive_text()
        start_payload = json.loads(start_message)
    except Exception:
        await websocket.send_json({"type": "error", "message": "Expected JSON start message"})
        await websocket.close(code=1003)
        return

    if start_payload.get("type") != "start":
        await websocket.send_json({"type": "error", "message": "Expected start message"})
        await websocket.close(code=1003)
        return

    if not is_valid_api_key(start_payload.get("api_key"), current_settings):
        await websocket.send_json({"type": "error", "message": "Invalid or missing API key"})
        await websocket.close(code=1008)
        return

    audio_format = start_payload.get("format", "pcm_s16le")
    if audio_format != "pcm_s16le":
        await websocket.send_json({"type": "error", "message": "Only pcm_s16le stream format is supported"})
        await websocket.close(code=1003)
        return

    sample_rate = int(start_payload.get("sample_rate", 16000))
    language = start_payload.get("language")
    min_chunk_bytes = max(1, int(sample_rate * 2 * current_settings.asr_stream_chunk_seconds))
    buffer = bytearray()
    segments: list[str] = []
    await websocket.send_json({"type": "ready"})

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                buffer.extend(message["bytes"])
                if len(buffer) >= min_chunk_bytes:
                    segment_text = _transcribe_pcm_chunk(bytes(buffer), sample_rate, language)
                    buffer.clear()
                    if segment_text:
                        segments.append(segment_text)
                        await websocket.send_json({"type": "partial", "text": " ".join(segments)})
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                if payload.get("type") == "end":
                    if buffer:
                        segment_text = _transcribe_pcm_chunk(bytes(buffer), sample_rate, language)
                        if segment_text:
                            segments.append(segment_text)
                    await websocket.send_json({"type": "final", "text": " ".join(segments)})
                    await websocket.close(code=1000)
                    return
            elif message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


def _transcribe_pcm_chunk(pcm_bytes: bytes, sample_rate: int, language: str | None) -> str:
    temp_path = write_pcm_s16le_wav(pcm_bytes, sample_rate)
    try:
        result = asr_transcriber.transcribe(temp_path, language=language)
        return result.text
    finally:
        remove_file(temp_path)
