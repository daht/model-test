from __future__ import annotations

import json
import time
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
    stateful_stable_commit_enabled = (
        current_settings.asr_stream_mode == "stateful" and current_settings.asr_stable_commit_enabled
    )
    committer = SentenceCommitter(
        commit_on_punctuation=(
            current_settings.asr_commit_on_punctuation and not stateful_stable_commit_enabled
        ),
    )
    silence_detector = SilenceEndpointDetector(
        silence_seconds=current_settings.asr_vad_silence_seconds,
        rms_threshold=current_settings.asr_vad_rms_threshold,
    )

    try:
        if current_settings.asr_stream_mode == "stateful":
            stable_committer = StablePunctuationCommitter(
                enabled=current_settings.asr_stable_commit_enabled,
                stable_seconds=current_settings.asr_stable_commit_seconds,
                min_chars=current_settings.asr_stable_commit_min_chars,
                min_updates=current_settings.asr_stable_commit_min_updates,
            )
            await _run_stateful_transcribe_stream(
                websocket,
                language,
                sample_rate,
                committer,
                stable_committer,
                silence_detector,
            )
        else:
            await websocket.send_json({"type": "ready"})
            await _run_chunked_transcribe_stream(
                websocket,
                current_settings,
                language,
                sample_rate,
                committer,
                silence_detector,
            )
    except WebSocketDisconnect:
        return


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
