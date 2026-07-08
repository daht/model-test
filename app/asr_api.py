from __future__ import annotations

import json
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from app.asr import ASRTranscriber, create_asr_transcriber
from app.audio import remove_file, save_upload_to_tempfile, write_pcm_s16le_wav
from app.auth import is_valid_api_key, require_api_key
from app.config import Settings, get_settings
from app.schemas import ASRHealthResponse, TranscribeResponse, TranscribeStreamInfoResponse

settings = get_settings()
asr_transcriber = create_asr_transcriber(settings)

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

app = FastAPI(
    title="Qwen ASR REST API",
    version="0.1.0",
    description="REST and WebSocket API template for Qwen3-ASR-1.7B deployment.",
)


def get_asr_transcriber() -> ASRTranscriber:
    return asr_transcriber


@dataclass
class SentenceCommitter:
    pending_text: str = ""
    confirmed_texts: list[str] = field(default_factory=list)

    def append(self, text: str) -> list[str]:
        if not text:
            return []

        self.pending_text = self._merge_recognition_text(text)
        committed, self.pending_text = _split_committed_sentences(self.pending_text)
        self.confirmed_texts.extend(committed)
        return committed

    def _merge_recognition_text(self, text: str) -> str:
        confirmed_text = "".join(self.confirmed_texts)
        current_text = confirmed_text + self.pending_text

        if confirmed_text and text.startswith(confirmed_text):
            return text[len(confirmed_text) :].lstrip()

        if current_text and text.startswith(current_text):
            return text[len(confirmed_text) :].lstrip()

        if self.pending_text and text.startswith(self.pending_text):
            return text

        return self.pending_text + text

    def commit_pending(self) -> str:
        sentence = self.pending_text.strip()
        if sentence:
            self.confirmed_texts.append(sentence)
        self.pending_text = ""
        return sentence


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
            sentence_end = index + 1
            while sentence_end < len(text) and text[sentence_end] in SENTENCE_CLOSERS:
                sentence_end += 1

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
def transcribe_stream_info(current_settings: Settings = Depends(get_settings)) -> TranscribeStreamInfoResponse:
    return TranscribeStreamInfoResponse(
        websocket_url="/v1/transcribe/stream",
        audio_format={
            "format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "recommended_chunk_ms": "100-500",
            "vad_silence_seconds": current_settings.asr_vad_silence_seconds,
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
            {"type": "ready"},
            {"type": "partial", "text": "..."},
            {"type": "sentence_final", "text": "..."},
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
    committer = SentenceCommitter()
    silence_detector = SilenceEndpointDetector(
        silence_seconds=current_settings.asr_vad_silence_seconds,
        rms_threshold=current_settings.asr_vad_rms_threshold,
    )
    await websocket.send_json({"type": "ready"})

    try:
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
    except WebSocketDisconnect:
        return


async def _send_committed_and_partial(
    websocket: WebSocket,
    committer: SentenceCommitter,
    text: str,
) -> None:
    for sentence in committer.append(text):
        await websocket.send_json({"type": "sentence_final", "text": sentence})
    if committer.pending_text:
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
