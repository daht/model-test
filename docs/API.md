# HY-MT / Qwen-ASR API Documentation

## Base URLs

Replace these placeholder domains with your production API domains.

```text
Translation API: https://translate-api.example.com
ASR API:         https://asr-api.example.com
TTS API:         https://tts-api.example.com
```

Interactive Swagger docs:

```text
https://translate-api.example.com/docs
https://asr-api.example.com/docs
https://tts-api.example.com/docs
```

OpenAPI JSON:

```text
https://translate-api.example.com/openapi.json
https://asr-api.example.com/openapi.json
https://tts-api.example.com/openapi.json
```

## Authentication

HTTP APIs require this header:

```http
X-API-Key: <your-api-key>
```

ASR WebSocket streaming authenticates the HTTP upgrade with `X-API-Key`. TTS
WebSocket streaming authenticates the upgrade with `Authorization: Bearer
<your-api-key>` and also accepts `X-API-Key`. Do not put credentials in a
WebSocket JSON message.

Production `qwen` and `qwen_vllm` configuration fails closed unless `API_KEY`
is at least 32 characters and is not a known placeholder. Generate it with
`openssl rand -hex 32`. Explicit `ASR_BACKEND=mock` test configurations may use
dummy keys.
Shell examples use `DEPLOYED_API_KEY` for the generated deployment value.

## Health Checks

### Translation Health

```http
GET /health
```

Example:

```bash
curl https://translate-api.example.com/health
```

Response:

```json
{
  "status": "ok",
  "model": "HY-MT1.5-1.8B",
  "backend": "transformers"
}
```

### ASR Health

```bash
curl https://asr-api.example.com/health
```

Response:

```json
{
  "status": "ok",
  "model": "Qwen3-ASR-1.7B",
  "backend": "qwen_vllm"
}
```

`GET /health` is liveness only and does not prove that model loading succeeded. Use `GET /ready` for traffic admission. `/ready` returns 503 until the owner thread has warmed the model and the coordinator accepts work; a ready response includes active stream count, queue depth, and queued audio seconds.

### TTS Health

```bash
curl https://tts-api.example.com/health
```

Response:

```json
{
  "status": "ok",
  "model": "CosyVoice",
  "backend": "mock",
  "sample_rate": 24000
}
```

## Text Translation

```http
POST /v1/translate
Content-Type: application/json
X-API-Key: <your-api-key>
```

Request:

```json
{
  "source_lang": "en",
  "target_lang": "zh",
  "text": "The customer can check out using Apple Pay.",
  "glossary": {
    "check out": "结账",
    "Apple Pay": "Apple Pay"
  },
  "preserve_format": true
}
```

Response:

```json
{
  "translation": "客户可以使用 Apple Pay 结账。",
  "source_lang": "en",
  "target_lang": "zh",
  "model": "HY-MT1.5-1.8B"
}
```

curl:

```bash
curl -X POST https://translate-api.example.com/v1/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "source_lang": "en",
    "target_lang": "zh",
    "text": "The customer can check out using Apple Pay.",
    "glossary": {
      "check out": "结账",
      "Apple Pay": "Apple Pay"
    },
    "preserve_format": true
  }'
```

Common language codes:

```text
zh      Simplified Chinese
zh-Hant Traditional Chinese
en      English
ja      Japanese
ko      Korean
de      German
fr      French
es      Spanish
ru      Russian
ar      Arabic
```

## ASR File Upload

Use this endpoint for offline transcription from a complete audio file. It is disabled by default on the live streaming instance and returns HTTP 503 with `detail.code=file_transcription_disabled`. Deploy a separate batch instance with `ASR_FILE_TRANSCRIBE_ENABLED=true`.

```http
POST /v1/transcribe
Content-Type: multipart/form-data
X-API-Key: <your-api-key>
```

Form fields:

```text
file      Required. wav/mp3/m4a/flac/ogg/webm audio file.
language  Optional. Language hint, such as zh, en, yue.
```

Response:

```json
{
  "text": "客户可以使用 Apple Pay 结账。",
  "language": "Chinese",
  "model": "Qwen3-ASR-1.7B"
}
```

curl:

```bash
curl -X POST https://asr-api.example.com/v1/transcribe \
  -H "X-API-Key: <your-api-key>" \
  -F "language=zh" \
  -F "file=@/path/to/audio.wav"
```

## ASR WebSocket Streaming

Use this endpoint for near-real-time transcription.

```text
wss://asr-api.example.com/v1/transcribe/stream
```

Inspect the active streaming mode and stateful settings:

```bash
curl https://asr-api.example.com/v1/transcribe/stream-info
```

Audio format:

```text
sample_rate: 16000
channels:    1
format:      pcm_s16le
chunk size:  100ms to 500ms recommended
endpointing: pinned Silero VAD v6.2.1 on CPU with onset/offset hysteresis
speech/silence: 250ms minimum speech, 800ms minimum trailing silence, 160ms hangover
pre-roll: 200ms rolling audio plus every onset-candidate frame
immutable commits: VAD endpoint, explicit segment, 30s utterance boundary, or end-of-input only
```

Production stateful qwen vLLM mode:

The image contract is `qwen-asr 0.0.6`, `vLLM 0.14.0`, ONNX Runtime 1.23.2,
and Silero VAD v6.2.1 with SHA256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
VAD runs only on `CPUExecutionProvider` with bounded intra/inter-op threads.

The Qwen weights have no checksum invented in this repository. Production
deployment requires an operator-approved JSON manifest created on a trusted
staging host with `python3 -m app.asr_artifacts create`, delivered alongside
the model through the release channel. Startup verifies the exact local file
set, sizes, SHA256 values, and absence of symlinks before loading Qwen. The
operator's approval and authenticated delivery of that manifest remain an
external provenance gate.

```bash
ASR_BACKEND=qwen_vllm
ASR_STREAM_MODE=stateful
ASR_REQUIRE_MODEL_MANIFEST=true
ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B.manifest.json
ASR_STREAM_CHUNK_SECONDS=1.0
ASR_MAX_UTTERANCE_SECONDS=30.0
ASR_STATE_WATCHDOG_SECONDS=120.0
ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8
ASR_VLLM_MAX_MODEL_LEN=65536
ASR_VLLM_MAX_NEW_TOKENS=32
ASR_STREAM_UNFIXED_CHUNK_NUM=2
ASR_STREAM_UNFIXED_TOKEN_NUM=5
ASR_VAD_MIN_SPEECH_MS=250
ASR_VAD_MIN_SILENCE_MS=800
ASR_VAD_HANGOVER_MS=160
ASR_VAD_PRE_ROLL_MS=200
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh build
```

Set `ASR_BACKEND=qwen` and `ASR_STREAM_MODE=chunked` to use the original chunked fallback.

Authenticate the WebSocket upgrade with `X-API-Key`. The first message after
the authenticated upgrade must be JSON:

```json
{
  "type": "start",
  "language": "zh",
  "sample_rate": 16000,
  "format": "pcm_s16le"
}
```

Server ready response:

```json
{
  "type": "ready",
  "sequence": 1
}
```

Then the client sends binary PCM chunks.

Server partial response:

```json
{
  "type": "partial",
  "text": "今天我们来介绍",
  "sequence": 2
}
```

`partial` is unconfirmed streaming text and may be replaced immediately by later `partial` messages, including revisions that add or remove punctuation. It only contains the current uncommitted tail, not previously committed text.

Server committed sentence response:

```json
{
  "type": "sentence_final",
  "text": "今天我们来介绍这个产品。",
  "sequence": 3
}
```

`sentence_final` text is confirmed and will not change. Stateful punctuation or repeated snapshots never make text immutable. A VAD endpoint, explicit `segment`, or the 30-second normal utterance boundary first flushes the official Qwen state, applies its final segment snapshot, emits at most one non-empty `sentence_final`, then clears `partial`.

End the stream:

```json
{
  "type": "finish"
}
```

Finalize the pending server audio buffer without closing the connection:

```json
{
  "type": "segment"
}
```

Use `segment` when the client needs an explicit utterance boundary without
closing the socket. Stateful mode flushes and finalizes the official Qwen
state. Chunked mode atomically transcribes any remainder below the normal chunk
threshold, appends it, emits `sentence_final` and the clearing `partial`, and
then clears the buffer. Both modes finalize a non-empty segment exactly once;
an empty boundary is valid.

Server final response:

```json
{
  "type": "final",
  "text": "剩余未确句文本",
  "sequence": 8
}
```

`final` is sent only after `finish` and contains the remaining uncommitted text. It does not repeat text already delivered by `sentence_final`. The completed transcript is all `sentence_final` messages in order plus the `final` text.

Every version 2 event, including `ready` and `error`, has a strictly increasing `sequence`. Gateway error codes emitted by the current Compose service are `invalid_start`, `invalid_audio`, `audio_limit`, `invalid_command`, `idle_timeout`, `session_timeout`, `audio_lag`, `backend_error`, `result_conflict`, and `overloaded`. Invalid PCM returns `invalid_audio` and closes the connection with code 1008. `invalid_start`, `invalid_audio`, `audio_limit`, and `invalid_command` close with 1008. `idle_timeout`, `session_timeout`, `audio_lag`, `backend_error`, and `result_conflict` close with 1011. `overloaded` closes with 1013. An invalid or missing upgrade key is rejected with 1008 before the WebSocket is accepted and therefore has no error event. A successful `finish` or `abort` closes with 1000.

The server limits active streams, queue jobs, queued audio seconds, per-connection lag, frame bytes, idle time, session time, and cumulative audio duration. It never accepts unbounded real-time backlog.

The conservative transport defaults are `ASR_MAX_FRAME_BYTES=16000` and
`ASR_WS_MAX_QUEUE=4`, which permit at most two seconds of 16 kHz PCM in the
Uvicorn receive queue. Startup rejects configurations where this transport
buffer exceeds `ASR_MAX_CONNECTION_LAG_SECONDS`. The absolute session deadline
covers model inference, server-side result processing, and transcript event
emission, so an expired session cannot publish a late `partial`.

One GPU must be owned by one ASR process with one Uvicorn worker. Horizontal scaling requires one service instance per GPU.

Test client:

```bash
API_KEY="$DEPLOYED_API_KEY" \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url wss://asr-api.example.com/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

`--print-mode display` renders the client-visible transcript as all confirmed `sentence_final` text plus the latest replaceable `partial` or `final` tail. For deployment smoke checks, set `EXPECT_ASR_STREAM_MODE=stateful`, `EXPECT_ASR_BACKEND=qwen_vllm`, and `EXPECT_ASR_STABLE_COMMIT_ENABLED=false` when running `scripts/smoke_asr.sh`. The smoke sends real non-silent PCM and requires continuous sequences, exactly one `final`, and a normal close.

## TTS Synthesis

Use this endpoint for non-streaming text-to-speech. The response body is WAV audio.

```http
POST /v1/tts
Content-Type: application/json
X-API-Key: <your-api-key>
Accept: audio/wav
```

Request:

```json
{
  "text": "你好，欢迎使用我们的产品。",
  "voice": "default"
}
```

Response:

```http
200 OK
Content-Type: audio/wav
```

curl:

```bash
curl -X POST https://tts-api.example.com/v1/tts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"text":"你好，欢迎使用我们的产品。","voice":"default"}' \
  --output output.wav
```

## TTS WebSocket Streaming

Use this endpoint when the client wants to send text incrementally and receive audio chunks.

```text
wss://tts-api.example.com/v1/tts/stream
```

Authenticate the WebSocket upgrade with `Authorization: Bearer <your-api-key>`.
The server first returns `connected_success`. Start the task with JSON:

```json
{
  "event": "task_start",
  "model": "Fun-CosyVoice3-0.5B-2512",
  "voice_setting": {"voice_id": "default"},
  "audio_setting": {
    "sample_rate": 24000,
    "format": "pcm",
    "channel": 1
  },
  "stream_options": {"audio_transport": "hex"}
}
```

The server responds with `task_started`. Send one or more text events:

```json
{"event":"task_continue","text":"你好，欢迎使用我们的产品。"}
```

Each `task_continued` response contains a PCM chunk as hex in
`data.audio`. Finish the input with:

```json
{"event":"task_finish"}
```

The server returns `task_finished` after all audio chunks, then closes normally.
For lower transport and decoding overhead, request binary mode:

```json
{"stream_options":{"audio_transport":"binary"}}
```

Each binary message is `TTS1` + little-endian uint32 chunk sequence +
little-endian uint64 sample offset + mono 24 kHz pcm_s16le. See
`docs/tts-a10-websocket-streaming-test-plan.md` for the complete protocol.

## Python Examples

### Translate

```python
import requests

api_key = "<your-api-key>"
url = "https://translate-api.example.com/v1/translate"

response = requests.post(
    url,
    headers={"X-API-Key": api_key},
    json={
        "source_lang": "en",
        "target_lang": "zh",
        "text": "The customer can check out using Apple Pay.",
        "preserve_format": True,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["translation"])
```

### ASR Upload

```python
import requests

api_key = "<your-api-key>"
url = "https://asr-api.example.com/v1/transcribe"

with open("/path/to/audio.wav", "rb") as audio:
    response = requests.post(
        url,
        headers={"X-API-Key": api_key},
        data={"language": "zh"},
        files={"file": ("audio.wav", audio, "audio/wav")},
        timeout=600,
    )

response.raise_for_status()
print(response.json()["text"])
```

### ASR WebSocket

```python
import asyncio
import json
import subprocess

import websockets


async def main():
    api_key = "<your-api-key>"
    audio_file = "/path/to/audio.wav"
    url = "wss://asr-api.example.com/v1/transcribe/stream"

    async with websockets.connect(
        url,
        max_size=None,
        additional_headers={"X-API-Key": api_key},
    ) as websocket:
        await websocket.send(json.dumps({
            "type": "start",
            "language": "zh",
            "sample_rate": 16000,
            "format": "pcm_s16le",
        }))
        print(await websocket.recv())

        process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", audio_file,
                "-ac", "1",
                "-ar", "16000",
                "-f", "s16le",
                "-"
            ],
            stdout=subprocess.PIPE,
        )

        assert process.stdout is not None
        while chunk := process.stdout.read(6400):
            await websocket.send(chunk)

        await websocket.send(json.dumps({"type": "finish"}))

        async for message in websocket:
            print(message)


asyncio.run(main())
```

## JavaScript Examples

### Translate

```javascript
const response = await fetch("https://translate-api.example.com/v1/translate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": "<your-api-key>"
  },
  body: JSON.stringify({
    source_lang: "en",
    target_lang: "zh",
    text: "The customer can check out using Apple Pay.",
    preserve_format: true
  })
});

if (!response.ok) {
  throw new Error(await response.text());
}

const data = await response.json();
console.log(data.translation);
```

### ASR Upload

```javascript
const form = new FormData();
form.append("language", "zh");
form.append("file", fileInput.files[0]);

const response = await fetch("https://asr-api.example.com/v1/transcribe", {
  method: "POST",
  headers: {
    "X-API-Key": "<your-api-key>"
  },
  body: form
});

if (!response.ok) {
  throw new Error(await response.text());
}

const data = await response.json();
console.log(data.text);
```

### ASR WebSocket

The browser `WebSocket` API cannot directly set the custom `X-API-Key` header
on the HTTP upgrade, so a direct browser connection cannot use the ASR endpoint
as documented. Browser deployments need a trusted authentication proxy, a
WebSocket subprotocol, or a short-lived token mechanism. The trusted boundary
must validate that browser credential and have it translated to `X-API-Key` for
the upstream ASR upgrade. Do not put credentials in the ASR `start` JSON.

Browser microphone audio is usually Float32 PCM at the device sample rate. Convert it to `16kHz mono pcm_s16le` before sending.

### TTS HTTP

```python
import requests

api_key = "<your-api-key>"
url = "https://tts-api.example.com/v1/tts"

response = requests.post(
    url,
    headers={"X-API-Key": api_key},
    json={"text": "你好，欢迎使用我们的产品。", "voice": "default"},
    timeout=120,
)
response.raise_for_status()

with open("output.wav", "wb") as audio:
    audio.write(response.content)
```

## Error Responses

Invalid or missing API key:

```json
{
  "detail": "Invalid or missing API key"
}
```

Unsupported audio type:

```json
{
  "detail": "Unsupported audio file type"
}
```

Audio too large:

```json
{
  "detail": "Audio file is too large"
}
```

TTS backend unavailable:

```json
{
  "detail": "CosyVoice backend is unavailable: ..."
}
```

WebSocket error:

```json
{
  "type": "error",
  "message": "Invalid or missing API key"
}
```

## Operational Notes

- First request can be slower because the model loads lazily.
- Translation service listens on port `8000`.
- ASR service listens on port `8002`.
- TTS service listens on port `8003`.
- For production, use HTTPS and a domain name in front of these services.
- Do not expose real API keys in client-side code for public websites.
