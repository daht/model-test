from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect

from app.auth import is_valid_api_key, require_api_key
from app.config import Settings, get_settings
from app.schemas import TTSHealthResponse, TTSInfoResponse, TTSRequest
from app.tts import TTSSynthesizer, create_tts_synthesizer

settings = get_settings()
tts_synthesizer = create_tts_synthesizer(settings)

app = FastAPI(
    title="CosyVoice TTS REST API",
    version="0.1.0",
    description="REST and WebSocket API template for CosyVoice text-to-speech deployment.",
)


def get_tts_synthesizer() -> TTSSynthesizer:
    return tts_synthesizer


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
        audio_format={
            "format": "wav",
            "sample_rate": current_settings.tts_sample_rate,
            "channels": 1,
        },
        start_message={
            "type": "start",
            "api_key": "<your-api-key>",
            "voice": current_settings.tts_default_voice,
            "sample_rate": current_settings.tts_sample_rate,
            "format": "wav",
        },
        text_message={"type": "text", "text": "..."},
        end_message={"type": "end"},
        server_messages=[
            {"type": "ready"},
            {"type": "done"},
            {"type": "error", "message": "..."},
        ],
    )


@app.post(
    "/v1/tts",
    dependencies=[Depends(require_api_key)],
)
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


@app.websocket("/v1/tts/stream")
async def synthesize_tts_stream(websocket: WebSocket) -> None:
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

    if start_payload.get("format", "wav") != "wav":
        await websocket.send_json({"type": "error", "message": "Only wav stream format is supported"})
        await websocket.close(code=1003)
        return

    voice = start_payload.get("voice") or current_settings.tts_default_voice
    await websocket.send_json({"type": "ready"})

    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    payload = {"type": "text", "text": message["text"]}

                message_type = payload.get("type")
                if message_type == "end":
                    await websocket.send_json({"type": "done"})
                    await websocket.close(code=1000)
                    return
                if message_type != "text":
                    await websocket.send_json({"type": "error", "message": "Expected text or end message"})
                    continue

                text = str(payload.get("text", ""))
                try:
                    _validate_text_length(text, current_settings)
                    audio = tts_synthesizer.synthesize(text, voice=voice)
                except (HTTPException, RuntimeError, ValueError) as exc:
                    message_text = exc.detail if isinstance(exc, HTTPException) else str(exc)
                    await websocket.send_json({"type": "error", "message": message_text})
                    continue

                await websocket.send_bytes(audio)
            elif message.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


def _validate_text_length(text: str, settings: Settings) -> None:
    if not text.strip():
        raise HTTPException(status_code=422, detail="text cannot be blank")

    if len(text) > settings.tts_max_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds maximum length of {settings.tts_max_text_chars} characters",
        )
