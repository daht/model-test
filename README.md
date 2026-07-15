# HY-MT1.5-1.8B REST API Deployment Template

## Semantic ASR Gateway

The ASR service starts `app.asr_gateway:app`. This semantic Gateway owns
authentication, sessions, aligned PCM buffering, chunk cursors, deadline
scheduling, sticky backend selection, and normalized `ready`, `partial`,
`sentence_final`, `final`, and `error` events. It is not a transparent proxy.
The service uses one Uvicorn worker and one local model owner per GPU.

Initial scheduling controls are environment variables:

```text
ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS=20
ASR_GATEWAY_MAX_READY_JOBS=64
ASR_GATEWAY_MAX_QUEUED_AUDIO_SECONDS=8
ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=4
ASR_GATEWAY_DEFAULT_UPDATE_MS=1500
ASR_GATEWAY_DRAIN_TIMEOUT_SECONDS=30
ASR_GATEWAY_DEFAULT_BACKEND=local
```

Runtime credentials are supplied through environment variables only. Backend
inventory and metrics require `X-API-Key` authentication. The local Qwen
adapter advertises serial stateful dispatch; cross-session dynamic batching is
enabled only for adapters that implement and advertise it. The pinned
`qwen-asr==0.0.6` / `vllm==0.14.0` stateful path uses
`Qwen/Qwen3-ASR-1.7B`, not the `-hf` Transformers export.

Model replacement follows drain-and-switch: warm and register the replacement,
route new sessions to it, drain the old worker for a bounded interval, abort any
remaining sessions explicitly, then unload it. A10 throughput, accuracy,
high-concurrency behavior, and production readiness remain external gates.

This repository deploys translation, Qwen3-ASR, and CosyVoice as separate FastAPI services.

The recommended target is NVIDIA A10 24GB. The live ASR service uses one process and one Uvicorn worker per GPU; do not increase `--workers` on a single GPU.

## Stateful ASR Production Contract

The production path is `ASR_BACKEND=qwen_vllm` with `ASR_STREAM_MODE=stateful`. All model loading and calls run on one owner thread behind a bounded coordinator, so synchronous GPU work does not block the FastAPI event loop. `/health` is liveness only; `/ready` returns 200 only after the pinned Silero VAD asset loads and Qwen completes a real non-silent streaming decode warmup.

Live streaming defaults to `ASR_FILE_TRANSCRIBE_ENABLED=false` because an in-progress file transcription cannot be preempted. Run file transcription on a separate batch ASR instance. Stateful punctuation and repeated text are always replaceable; only a VAD endpoint, explicit `segment`, the exact maximum-utterance boundary, or `finish` can freeze text.

Protocol version 2 guarantees monotonically increasing `sequence` values. Append `sentence_final` events permanently and replace the displayed tail with every `partial` or `final`. A commit is followed by `partial: ""` when no unconfirmed tail remains.

The default WebSocket transport accepts frames up to `ASR_MAX_FRAME_BYTES=16000`
(0.5 seconds of PCM) and queues at most `ASR_WS_MAX_QUEUE=4` frames. Settings
validation requires their worst-case buffered audio to fit within
`ASR_MAX_CONNECTION_LAG_SECONDS`; change all three values together. Shutdown
stops admission immediately and waits at most `ASR_SHUTDOWN_GRACE_SECONDS` for
an already-running model call.

The ASR image pins `qwen-asr[vllm]==0.0.6`, `vllm==0.14.0`, and
`onnxruntime==1.23.2`. It downloads Silero VAD v6.2.1 only during image build,
verifies SHA256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`,
and includes the upstream MIT license. Runtime startup never downloads VAD assets.

## Files

- `app/main.py`: FastAPI routes, health check, API key guard.
- `app/asr_api.py`: Qwen3-ASR file upload and WebSocket routes.
- `app/asr.py`: Qwen3-ASR lazy-loading inference wrappers, including chunked and qwen vLLM stateful streaming backends.
- `app/tts_api.py`: CosyVoice HTTP and WebSocket TTS routes.
- `app/tts.py`: CosyVoice lazy-loading inference wrapper and mock WAV backend.
- `app/model.py`: mock translator for tests and Transformers translator for production.
- `app/schemas.py`: request and response models.
- `Dockerfile`: GPU-capable image based on NVIDIA CUDA runtime.
- `Dockerfile.asr`: Qwen3-ASR image with optional qwen-asr vLLM dependencies.
- `requirements-asr-vllm.txt`: optional dependency set for stateful Qwen3-ASR vLLM streaming.
- `docker-compose.yml`: local or cloud VM deployment.
- `nginx.conf.example`: reverse proxy starter config.
- `scripts/bootstrap_ubuntu_gpu.sh`: Ubuntu GPU server bootstrap script.
- `scripts/deploy_remote.sh`: rsync and remote Docker Compose deployment.
- `scripts/smoke_test.sh`: post-deploy API check.
- `scripts/smoke_asr.sh`: post-deploy Qwen3-ASR API check.
- `scripts/deploy_asr_cloud.sh`: fail-closed release, exact-image cutover, live verification, and rollback for the cloud ASR service.
- `scripts/verify_asr_release.sh`: layered commit, release, and live ASR verification. See `docs/asr-release-verification.md`.
- `scripts/update_service.sh`: update/recreate the cloud Docker service.
- `cloud/README-A10.md`: A10-specific deployment runbook.

## Deploy to Your A10 Cloud Server

Assumptions:

- Server OS: Ubuntu 22.04 LTS
- GPU: NVIDIA A10 24GB
- SSH user can run `sudo`
- The firewall/security group allows inbound `22` and `8000`

On the server, install Docker, NVIDIA driver, and NVIDIA Container Toolkit:

```bash
sudo bash scripts/bootstrap_ubuntu_gpu.sh
```

If the script installs the NVIDIA driver, reboot once:

```bash
sudo reboot
```

Then run the bootstrap script again and verify GPU Docker access:

```bash
sudo bash scripts/bootstrap_ubuntu_gpu.sh
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Prepare config:

```bash
cp cloud/A10.env.example .env
openssl rand -hex 32
```

Put the generated secret into `.env` as `API_KEY`. The `qwen` and `qwen_vllm`
backends refuse to start with a missing key, a key shorter than 32 characters,
or a known example value. Explicit `mock` backends continue to accept dummy
keys for local tests.
Client command examples use `DEPLOYED_API_KEY` for that generated value.

If your model weights are local, copy them into:

```text
models/HY-MT1.5-1.8B
models/Qwen3-ASR-1.7B
models/CosyVoice
models/CosyVoice-ttsfrd
```

Qwen model provenance is a target-host deployment gate. The release operator
must choose an approved immutable upstream revision, download it on a trusted
staging host, and create the approval manifest there:

```bash
hf download Qwen/Qwen3-ASR-1.7B \
  --revision "$APPROVED_QWEN_REVISION" \
  --local-dir models/Qwen3-ASR-1.7B
python3 -m app.asr_artifacts create \
  --model-dir models/Qwen3-ASR-1.7B \
  --output models/Qwen3-ASR-1.7B.manifest.json \
  --source Qwen/Qwen3-ASR-1.7B \
  --revision "$APPROVED_QWEN_REVISION"
```

Transfer the model and manifest to the target through the approved release
channel. Do not create the approval manifest from an unverified target-host
download: that records hashes but establishes no provenance. At Qwen startup,
the service rejects missing, extra, changed, or symlinked artifacts when
`ASR_REQUIRE_MODEL_MANIFEST=true`. Repository tests cannot supply the accepted
revision or manifest because the production weights and release approval are
external artifacts.

After `.env`, the approved model/manifest, and external Chinese/Japanese live
speech inputs exist, deploy ASR with the one-command wrapper:

```bash
export ASR_LIVE_ZH_AUDIO=/secure/release-input/chinese-speech.flac
export ASR_LIVE_JA_AUDIO=/secure/release-input/japanese-speech.flac
export ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS=10
export ASR_LIVE_MAX_GPU_MEMORY_MIB=23000
read -rsp "Deployed ASR API key: " ASR_LIVE_API_KEY; echo
export ASR_LIVE_API_KEY
scripts/deploy_asr_cloud.sh
unset ASR_LIVE_API_KEY
```

The thresholds must come from the approved service SLO and GPU headroom policy.
The wrapper defaults to full release/deploy/live verification, retains evidence
and rollback backups below `/secure`, and restores the previous image,
configuration, and approved manifest after a post-cutover failure. Inspect the
workflow with `scripts/deploy_asr_cloud.sh --dry-run`; `--skip-live` is an
explicit reduced gate and does not establish live verification.

The one-command workflow has planned downtime on a single GPU. It validates and
receipts the running rollback baseline, stops that model owner, and only then
runs release R08. After cutover it uses the receipt-bound `deployed-live` layer,
which runs L01-L04 without building, warming, or starting another Qwen owner.

CosyVoice also needs the official runtime code in the build context:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
```

Keep this A10 default config:

```dotenv
MODEL_BACKEND=transformers
MODEL_TASK=causal-lm
DEVICE=auto
TORCH_DTYPE=float16
MAX_NEW_TOKENS=1024
ASR_MODEL_ID=/models/Qwen3-ASR-1.7B
ASR_REQUIRE_MODEL_MANIFEST=true
ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B.manifest.json
ASR_BACKEND=qwen_vllm
ASR_STREAM_MODE=stateful
ASR_STREAM_CHUNK_SECONDS=1.0
ASR_MAX_UTTERANCE_SECONDS=30.0
ASR_STATE_WATCHDOG_SECONDS=120.0
ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8
ASR_VLLM_MAX_MODEL_LEN=65536
ASR_VLLM_MAX_NEW_TOKENS=32
ASR_STREAM_UNFIXED_CHUNK_NUM=2
ASR_STREAM_UNFIXED_TOKEN_NUM=5
ASR_VAD_MODEL_VERSION=6.2.1
ASR_VAD_MODEL_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
ASR_VAD_ONSET_THRESHOLD=0.65
ASR_VAD_OFFSET_THRESHOLD=0.35
ASR_VAD_MIN_SPEECH_MS=250
ASR_VAD_MIN_SILENCE_MS=800
ASR_VAD_HANGOVER_MS=160
ASR_VAD_PRE_ROLL_MS=200
TTS_MODEL_ID=/models/CosyVoice
TTS_BACKEND=cosyvoice
TTS_COSYVOICE_REPO=/opt/CosyVoice
TTS_SAMPLE_RATE=24000
```

Use `ASR_BACKEND=qwen` and `ASR_STREAM_MODE=chunked` if you need the original temp-WAV chunked fallback instead of stateful qwen vLLM streaming.

If HY-MT loads as an encoder-decoder model in your weights, change:

```dotenv
MODEL_TASK=seq2seq-lm
```

Start service:

```bash
docker compose up --build -d
docker compose logs -f hy-mt-api
docker compose logs -f qwen-asr-api
docker compose logs -f cosyvoice-tts-api
```

Update service after code changes:

```bash
cd /opt/model-test
scripts/update_service.sh
```

Update both translation and ASR services:

```bash
cd /opt/model-test
scripts/update_all_services.sh
```

The default update path pulls directly from the official GitHub remote:

```bash
cd /opt/model-test
scripts/update_service.sh
```

If GitHub is temporarily unstable, you can still push code from your local machine as a fallback:

```bash
REMOTE_HOST=ubuntu@your-server-ip REMOTE_DIR=/opt/model-test ENV_FILE=.env scripts/deploy_remote.sh
```

Reload only `.env` changes:

```bash
scripts/update_service.sh env
```

Restart after replacing model files:

```bash
scripts/update_service.sh restart
```

Update only the ASR container:

```bash
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh
```

Rebuild the ASR image after changing stateful vLLM dependencies:

```bash
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh build
```

Verify:

```bash
API_KEY="$DEPLOYED_API_KEY" BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
API_KEY="$DEPLOYED_API_KEY" BASE_URL=http://your-server-ip:8002 scripts/smoke_asr.sh
curl -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，欢迎使用。"}' \
  --output output.wav \
  http://your-server-ip:8003/v1/tts
```

Remote deploy from your local machine is also supported:

```bash
REMOTE_HOST=root@your-server-ip ENV_FILE=.env scripts/deploy_remote.sh
API_KEY="$DEPLOYED_API_KEY" BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
```

## Quick Start

Create the environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Set API_KEY to the previously generated value.
MODEL_ID=/models/HY-MT1.5-1.8B
MODEL_BACKEND=transformers
MODEL_TASK=causal-lm
```

If the model is on Hugging Face, set `MODEL_ID` to the model id. If the model is local, put the weights under `./models/HY-MT1.5-1.8B`.

For Qwen3-ASR, use the approved immutable revision and trusted-staging manifest
procedure above. The target layout is:

```text
models/Qwen3-ASR-1.7B/
models/Qwen3-ASR-1.7B.manifest.json
```

For CosyVoice TTS, download the model and runtime resources:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/CosyVoice
hf download FunAudioLLM/CosyVoice-ttsfrd --local-dir models/CosyVoice-ttsfrd
```

Run with Docker Compose:

```bash
docker compose up --build -d
```

Check health:

```bash
curl http://localhost:8000/health
curl http://localhost:8002/health
curl http://localhost:8002/ready
curl http://localhost:8003/health
```

Translate:

```bash
curl -X POST http://localhost:8000/v1/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-production-api-key" \
  -d '{
    "source_lang": "zh",
    "target_lang": "en",
    "text": "你好，欢迎使用我们的产品。",
    "glossary": {
      "产品": "product"
    },
    "preserve_format": true
  }'
```

Transcribe an audio file on a batch instance configured with `ASR_FILE_TRANSCRIBE_ENABLED=true`:

```bash
curl -X POST http://localhost:8002/v1/transcribe \
  -H "X-API-Key: your-production-api-key" \
  -F "language=zh" \
  -F "file=@/path/to/audio.wav"
```

Synthesize speech:

```bash
curl -X POST http://localhost:8003/v1/tts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-production-api-key" \
  -d '{"text":"你好，欢迎使用我们的产品。","voice":"default"}' \
  --output output.wav
```

WebSocket streaming protocol:

```text
WS ws://localhost:8002/v1/transcribe/stream
```

The WebSocket upgrade must include `X-API-Key`. After the authenticated upgrade,
the client sends a JSON start message:

```json
{
  "type": "start",
  "language": "zh",
  "sample_rate": 16000,
  "format": "pcm_s16le"
}
```

Then the client sends binary PCM chunks. The server returns:

```json
{"type":"ready","sequence":1}
{"type":"partial","text":"...","sequence":2}
{"type":"sentence_final","text":"...","sequence":3}
{"type":"partial","text":"","sequence":4}
{"type":"final","text":"...","sequence":5}
```

`partial` is the current segment snapshot and is replaceable immediately, including revisions that add or remove model-generated punctuation. Gateway-owned Silero VAD uses onset/offset hysteresis, a 250ms minimum speech duration, 800ms trailing silence, 160ms hangover, and a 200ms rolling pre-roll. Silence/noise and short bursts are explicitly discarded by the Gateway. On `finish`, `final` contains only the remaining unconfirmed segment, so clients should render:

```text
display_text = all sentence_final text in order + latest partial or final tail
```

End the stream with:

```json
{"type":"finish"}
```

Finalize pending audio while keeping the WebSocket session open:

```json
{"type":"segment"}
```

In both streaming modes, `segment` first transcribes any buffered PCM, appends
the returned text, emits `sentence_final` followed by the clearing `partial`
when the segment is non-empty, and only then resets the segment. An empty
buffer is a valid no-op boundary.

You can test the ASR stream with the included client:

```bash
API_KEY="$DEPLOYED_API_KEY" \
python scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

The client converts the input audio to `16kHz mono pcm_s16le` with `ffmpeg`, sends 200ms chunks, and prints either raw `partial` / `sentence_final` / `final` events or, with `--print-mode display`, the rendered transcript as confirmed text plus the latest replaceable partial.

Check the server-selected ASR streaming mode before testing:

```bash
curl -H "X-API-Key: ${API_KEY}" http://localhost:8002/v1/transcribe/stream-info
```

Stateful qwen vLLM streaming uses these production settings:

```bash
ASR_BACKEND=qwen_vllm
ASR_STREAM_MODE=stateful
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

Smoke-test the expected ASR mode:

```bash
API_KEY="$DEPLOYED_API_KEY" \
EXPECT_ASR_STREAM_MODE=stateful \
EXPECT_ASR_BACKEND=qwen_vllm \
EXPECT_ASR_STABLE_COMMIT_ENABLED=false \
BASE_URL=http://127.0.0.1:8002 \
scripts/smoke_asr.sh
```

TTS WebSocket streaming protocol:

```text
WS ws://localhost:8003/v1/tts/stream
```

Client sends a JSON start message:

```json
{
  "type": "start",
  "api_key": "your-production-api-key",
  "voice": "default",
  "sample_rate": 24000,
  "format": "wav"
}
```

The server returns `{"type":"ready"}`. Send text messages as either plain text or JSON:

```json
{"type":"text","text":"你好，欢迎使用我们的产品。"}
```

For each text message, the server returns one binary WAV audio chunk. End the stream with:

```json
{"type":"end"}
```

The server sends `{"type":"done"}` and closes the WebSocket.

## Local Test Mode

Use the mock backend when you only want to verify the API service:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
MODEL_BACKEND=mock ASR_BACKEND=mock TTS_BACKEND=mock API_KEY=test-key .venv/bin/python -m pytest tests/test_api.py -q
MODEL_BACKEND=mock ASR_BACKEND=mock TTS_BACKEND=mock API_KEY=test-key .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Cloud VM Notes

Install these on the GPU VM:

- NVIDIA Driver
- Docker
- NVIDIA Container Toolkit

Then deploy:

```bash
cp .env.example .env
docker compose up --build -d
```

For HTTPS, put Nginx or a cloud load balancer in front of port `8000`. Use `nginx.conf.example` as the starting reverse proxy config, then add TLS with Certbot or your cloud provider certificate manager.

## API

`GET /health`

Response:

```json
{
  "status": "ok",
  "model": "HY-MT1.5-1.8B",
  "backend": "transformers"
}
```

`POST /v1/translate`

Headers:

```text
X-API-Key: your-production-api-key
```

Request:

```json
{
  "source_lang": "zh",
  "target_lang": "en",
  "text": "你好，欢迎使用我们的产品。",
  "glossary": {
    "产品": "product"
  },
  "preserve_format": true
}
```

`POST /v1/transcribe`

Headers:

```text
X-API-Key: your-production-api-key
```

Multipart fields:

```text
file: wav/mp3/m4a/flac/ogg/webm
language: optional language hint, such as zh, en, yue
```

Response:

```json
{
  "text": "客户可以使用 Apple Pay 结账。",
  "language": "Chinese",
  "model": "Qwen3-ASR-1.7B"
}
```

Response:

```json
{
  "translation": "Hello, welcome to use our product.",
  "source_lang": "zh",
  "target_lang": "en",
  "model": "HY-MT1.5-1.8B"
}
```

## Production Checklist

- Generate `API_KEY` with `openssl rand -hex 32`; production ASR rejects short and known placeholder values.
- Supply the operator-approved Qwen manifest and keep `ASR_REQUIRE_MODEL_MANIFEST=true`.
- Use HTTPS in front of the service.
- Add request rate limiting at Nginx, API Gateway, or load balancer level.
- Mount model weights read-only.
- Monitor GPU memory, request latency, error rate, and container restarts.
- Tune `MAX_NEW_TOKENS`, batching, and GPU size after load testing.
