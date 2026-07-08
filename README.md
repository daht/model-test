# HY-MT1.5-1.8B REST API Deployment Template

This template deploys HY-MT1.5-1.8B behind a FastAPI REST service.

The recommended target for your current server is NVIDIA A10 24GB. Use `float16`, one GPU, and no quantization for the first production deployment.

## Files

- `app/main.py`: FastAPI routes, health check, API key guard.
- `app/asr_api.py`: Qwen3-ASR file upload and WebSocket routes.
- `app/asr.py`: Qwen3-ASR lazy-loading inference wrapper and mock backend.
- `app/model.py`: mock translator for tests and Transformers translator for production.
- `app/schemas.py`: request and response models.
- `Dockerfile`: GPU-capable image based on NVIDIA CUDA runtime.
- `docker-compose.yml`: local or cloud VM deployment.
- `nginx.conf.example`: reverse proxy starter config.
- `scripts/bootstrap_ubuntu_gpu.sh`: Ubuntu GPU server bootstrap script.
- `scripts/deploy_remote.sh`: rsync and remote Docker Compose deployment.
- `scripts/smoke_test.sh`: post-deploy API check.
- `scripts/smoke_asr.sh`: post-deploy Qwen3-ASR API check.
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

Put the generated secret into `.env` as `API_KEY`.

If your model weights are local, copy them into:

```text
models/HY-MT1.5-1.8B
models/Qwen3-ASR-1.7B-hf
```

Keep this A10 default config:

```dotenv
MODEL_BACKEND=transformers
MODEL_TASK=causal-lm
DEVICE=auto
TORCH_DTYPE=float16
MAX_NEW_TOKENS=1024
ASR_MODEL_ID=/models/Qwen3-ASR-1.7B-hf
ASR_BACKEND=qwen
ASR_TORCH_DTYPE=bfloat16
```

If HY-MT loads as an encoder-decoder model in your weights, change:

```dotenv
MODEL_TASK=seq2seq-lm
```

Start service:

```bash
docker compose up --build -d
docker compose logs -f hy-mt-api
docker compose logs -f qwen-asr-api
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

Verify:

```bash
API_KEY=your-api-key BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8002 scripts/smoke_asr.sh
```

Remote deploy from your local machine is also supported:

```bash
REMOTE_HOST=root@your-server-ip ENV_FILE=.env scripts/deploy_remote.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
```

## Quick Start

Create the environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
API_KEY=your-production-api-key
MODEL_ID=/models/HY-MT1.5-1.8B
MODEL_BACKEND=transformers
MODEL_TASK=causal-lm
```

If the model is on Hugging Face, set `MODEL_ID` to the model id. If the model is local, put the weights under `./models/HY-MT1.5-1.8B`.

For Qwen3-ASR, download the model to:

```bash
hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen3-ASR-1.7B-hf
```

Run with Docker Compose:

```bash
docker compose up --build -d
```

Check health:

```bash
curl http://localhost:8000/health
curl http://localhost:8002/health
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

Transcribe an audio file:

```bash
curl -X POST http://localhost:8002/v1/transcribe \
  -H "X-API-Key: your-production-api-key" \
  -F "language=zh" \
  -F "file=@/path/to/audio.wav"
```

WebSocket streaming protocol:

```text
WS ws://localhost:8002/v1/transcribe/stream
```

Client sends a JSON start message:

```json
{
  "type": "start",
  "api_key": "your-production-api-key",
  "language": "zh",
  "sample_rate": 16000,
  "format": "pcm_s16le"
}
```

Then the client sends binary PCM chunks. The server returns:

```json
{"type":"partial","text":"..."}
{"type":"sentence_final","text":"..."}
{"type":"final","text":"..."}
```

`partial` is the current unconfirmed streaming text and may change. `sentence_final` is a committed sentence and will not be sent again or changed by later messages. Sentences are committed when the unconfirmed text reaches a sentence-ending punctuation mark or when the server detects 1.0 second of continuous silence in the PCM stream. Later `partial` messages only contain the uncommitted tail after the latest committed sentence. On `end`, `final` contains only the remaining uncommitted text, so clients should render:

```text
display_text = all sentence_final text in order + latest partial or final tail
```

End the stream with:

```json
{"type":"end"}
```

Clear pending audio while keeping the WebSocket session open:

```json
{"type":"segment"}
```

You can test the stream with the included client:

```bash
API_KEY=your-production-api-key \
python scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh \
  --realtime
```

The client converts the input audio to `16kHz mono pcm_s16le` with `ffmpeg`, sends 200ms chunks, and prints `partial` / `sentence_final` / `final` messages.

## Local Test Mode

Use the mock backend when you only want to verify the API service:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
MODEL_BACKEND=mock API_KEY=test-key .venv/bin/python -m pytest tests/test_api.py -q
MODEL_BACKEND=mock API_KEY=test-key .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
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

- Replace `API_KEY` with a long random secret.
- Use HTTPS in front of the service.
- Add request rate limiting at Nginx, API Gateway, or load balancer level.
- Mount model weights read-only.
- Monitor GPU memory, request latency, error rate, and container restarts.
- Tune `MAX_NEW_TOKENS`, batching, and GPU size after load testing.
