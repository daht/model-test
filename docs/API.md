# HY-MT / Qwen-ASR API Documentation

## Base URLs

Replace these placeholder domains with your production API domains.

```text
Translation API: https://translate-api.example.com
ASR API:         https://asr-api.example.com
```

Interactive Swagger docs:

```text
https://translate-api.example.com/docs
https://asr-api.example.com/docs
```

OpenAPI JSON:

```text
https://translate-api.example.com/openapi.json
https://asr-api.example.com/openapi.json
```

## Authentication

HTTP APIs require this header:

```http
X-API-Key: <your-api-key>
```

WebSocket streaming sends the API key in the first JSON message.

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
  "backend": "qwen"
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

Use this endpoint for offline transcription from a complete audio file.

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

Audio format:

```text
sample_rate: 16000
channels:    1
format:      pcm_s16le
chunk size:  100ms to 500ms recommended
VAD commit:  1.0s continuous silence by default
```

First client message must be JSON:

```json
{
  "type": "start",
  "api_key": "<your-api-key>",
  "language": "zh",
  "sample_rate": 16000,
  "format": "pcm_s16le"
}
```

Server ready response:

```json
{
  "type": "ready"
}
```

Then the client sends binary PCM chunks.

Server partial response:

```json
{
  "type": "partial",
  "text": "今天我们来介绍"
}
```

`partial` is unconfirmed streaming text and may be replaced by later `partial` messages. It only contains the current uncommitted tail, not previously committed sentences.

Server committed sentence response:

```json
{
  "type": "sentence_final",
  "text": "今天我们来介绍这个产品。"
}
```

`sentence_final` text is confirmed and will not change. It is sent when the unconfirmed text reaches sentence-ending punctuation, or when the server detects 1.0 second of continuous silence in the PCM stream. Clients should append every `sentence_final` in order, then display the latest `partial` after that confirmed prefix.

End the stream:

```json
{
  "type": "end"
}
```

Clear the pending server audio buffer without closing the connection:

```json
{
  "type": "segment"
}
```

Use `segment` during long-running listening when the client wants to discard pending server audio without closing the socket. `segment` is not a commit command; sentence confirmation is triggered by punctuation or by the server's silence detector.

Server final response:

```json
{
  "type": "final",
  "text": "剩余未确句文本"
}
```

`final` is sent only after `end` and contains the remaining uncommitted text. It does not repeat text already delivered by `sentence_final`. The completed transcript is all `sentence_final` messages in order plus the `final` text.

Test client:

```bash
API_KEY=<your-api-key> \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url wss://asr-api.example.com/v1/transcribe/stream \
  --language zh \
  --realtime
```

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

    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "type": "start",
            "api_key": api_key,
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

        await websocket.send(json.dumps({"type": "end"}))

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

```javascript
const ws = new WebSocket("wss://asr-api.example.com/v1/transcribe/stream");

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "start",
    api_key: "<your-api-key>",
    language: "zh",
    sample_rate: 16000,
    format: "pcm_s16le"
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(message.type, message.text || message.message || "");
};

// After sending binary PCM chunks:
// ws.send(JSON.stringify({ type: "end" }));
```

Browser microphone audio is usually Float32 PCM at the device sample rate. Convert it to `16kHz mono pcm_s16le` before sending.

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
- For production, use HTTPS and a domain name in front of these services.
- Do not expose real API keys in client-side code for public websites.
