# HY-MT1.5-1.8B on NVIDIA A10

Recommended VM:

- GPU: 1 x NVIDIA A10, 24GB VRAM
- OS: Ubuntu 22.04 LTS
- Disk: 80GB minimum, 150GB+ recommended if storing model weights locally
- Open ports: `22`, `8000`; use `80/443` only if you add Nginx or a load balancer

## 1. Bootstrap the server

Upload this project to the server once, then run:

```bash
sudo bash scripts/bootstrap_ubuntu_gpu.sh
```

If the script installs the NVIDIA driver, reboot:

```bash
sudo reboot
```

After reconnecting:

```bash
sudo bash scripts/bootstrap_ubuntu_gpu.sh
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. Prepare model weights

Local model path option:

```bash
sudo mkdir -p /opt/hy-mt-api/models/HY-MT1.5-1.8B
sudo rsync -av /path/to/HY-MT1.5-1.8B/ /opt/hy-mt-api/models/HY-MT1.5-1.8B/
```

Hugging Face option:

```bash
MODEL_ID=your-org-or-user/HY-MT1.5-1.8B
```

Put that value in `.env`.

Qwen3-ASR model:

```bash
cd /opt/model-test
hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen3-ASR-1.7B-hf
```

CosyVoice TTS runtime and model:

```bash
cd /opt/model-test
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/CosyVoice
hf download FunAudioLLM/CosyVoice-ttsfrd --local-dir models/CosyVoice-ttsfrd
```

## 3. Configure A10 defaults

```bash
cp cloud/A10.env.example .env
openssl rand -hex 32
```

Use the generated value as `API_KEY`.

For A10, keep:

```bash
TORCH_DTYPE=float16
DEVICE=auto
MAX_NEW_TOKENS=1024
ASR_TORCH_DTYPE=bfloat16
ASR_DEVICE=auto
ASR_BACKEND=qwen_vllm
ASR_STREAM_MODE=stateful
ASR_STREAM_CHUNK_SECONDS=1.0
ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8
ASR_VLLM_MAX_NEW_TOKENS=32
ASR_STREAM_UNFIXED_CHUNK_NUM=2
ASR_STREAM_UNFIXED_TOKEN_NUM=5
ASR_VAD_SILENCE_SECONDS=0.8
ASR_STABLE_COMMIT_ENABLED=true
ASR_STABLE_COMMIT_SECONDS=1.0
ASR_STABLE_COMMIT_MIN_CHARS=8
ASR_STABLE_COMMIT_MIN_UPDATES=2
ASR_PROTOCOL_VERSION=2
ASR_EAGER_LOAD=true
ASR_FILE_TRANSCRIBE_ENABLED=false
ASR_MAX_ACTIVE_STREAMS=2
ASR_INFERENCE_QUEUE_SIZE=16
ASR_MAX_QUEUED_AUDIO_SECONDS=4.0
ASR_MAX_CONNECTION_LAG_SECONDS=2.0
ASR_MAX_FRAME_BYTES=16000
ASR_WS_MAX_QUEUE=4
ASR_START_TIMEOUT_SECONDS=10
ASR_IDLE_TIMEOUT_SECONDS=30
ASR_MAX_SESSION_SECONDS=1800
ASR_MAX_AUDIO_SECONDS=1800
ASR_STREAM_QUEUE_TIMEOUT_SECONDS=2.0
ASR_STREAM_INFERENCE_TIMEOUT_SECONDS=15.0
ASR_FILE_INFERENCE_TIMEOUT_SECONDS=300.0
ASR_SHUTDOWN_GRACE_SECONDS=10.0
TTS_BACKEND=cosyvoice
TTS_MODEL_ID=/models/CosyVoice
TTS_COSYVOICE_REPO=/opt/CosyVoice
TTS_SAMPLE_RATE=24000
```

Keep exactly one ASR process and one Uvicorn worker per GPU. `ASR_MAX_ACTIVE_STREAMS=2` is a conservative rollout setting, not a capacity claim. Calibrate it with 1, 2, 4, and 8 real-time streams while recording first-partial latency, queue wait p50/p95, inference p50/p95, RTF, GPU memory, and disconnect/error counts.

At 16 kHz PCM16, the default frame is 0.5 seconds and four Uvicorn queue slots
buffer at most two seconds of audio. Keep
`ASR_WS_MAX_QUEUE * ASR_MAX_FRAME_BYTES / 32000 <= ASR_MAX_CONNECTION_LAG_SECONDS`;
the application validates this relationship at startup.

The live instance intentionally disables file upload. Use a separate batch instance with `ASR_FILE_TRANSCRIBE_ENABLED=true`; otherwise a running file job can stall every live stream. Use `ASR_BACKEND=qwen` and `ASR_STREAM_MODE=chunked` only as a fallback.

If the model is encoder-decoder rather than causal language model, set:

```bash
MODEL_TASK=seq2seq-lm
```

## 4. Deploy

From the server:

```bash
docker compose up --build -d
docker compose logs -f hy-mt-api
docker compose logs -f qwen-asr-api
docker compose logs -f cosyvoice-tts-api
```

From your local machine:

```bash
REMOTE_HOST=root@your-server-ip ENV_FILE=.env scripts/deploy_remote.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8002 scripts/smoke_asr.sh
curl -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，欢迎使用。"}' \
  --output output.wav \
  http://your-server-ip:8003/v1/tts
```

## 5. Client request

```bash
curl -X POST http://your-server-ip:8000/v1/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
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

ASR upload:

```bash
curl -X POST http://your-server-ip:8002/v1/transcribe \
  -H "X-API-Key: your-api-key" \
  -F "language=zh" \
  -F "file=@/path/to/audio.wav"
```

ASR WebSocket endpoint:

```text
ws://your-server-ip:8002/v1/transcribe/stream
```

Stateful `partial` text is immediately replaceable. A punctuated prefix becomes `sentence_final` only after the exact prefix survives 1.0 second of additional processed audio and at least two updates, with a minimum of 8 non-whitespace characters. The A10 configuration uses 0.8 seconds of VAD silence as the fallback force-commit path.

Test WebSocket streaming from the server:

```bash
API_KEY=your-api-key \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

Test from your own machine by changing the URL:

```bash
API_KEY=your-api-key \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://your-server-ip:8002/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

Check the deployed ASR mode during smoke tests:

```bash
API_KEY=your-api-key \
EXPECT_ASR_STREAM_MODE=stateful \
EXPECT_ASR_BACKEND=qwen_vllm \
EXPECT_ASR_STABLE_COMMIT_ENABLED=true \
BASE_URL=http://127.0.0.1:8002 \
scripts/smoke_asr.sh
```

TTS HTTP endpoint:

```bash
curl -X POST http://your-server-ip:8003/v1/tts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text":"你好，欢迎使用。","voice":"default"}' \
  --output output.wav
```

## 6. Operations

```bash
docker compose ps
docker compose logs -f hy-mt-api
docker compose logs -f qwen-asr-api
docker compose logs -f cosyvoice-tts-api
docker stats
nvidia-smi
docker compose restart hy-mt-api
docker compose restart qwen-asr-api
docker compose restart cosyvoice-tts-api
docker compose pull && docker compose up --build -d
```

## 7. Updates

Run this from the project directory on the server:

```bash
cd /opt/model-test
scripts/update_service.sh
```

Update both translation and ASR services:

```bash
cd /opt/model-test
scripts/update_all_services.sh
```

By default, update directly from the official GitHub remote:

```bash
cd /opt/model-test
scripts/update_service.sh
```

If GitHub is temporarily unstable, push code from your local machine as a fallback:

```bash
REMOTE_HOST=ubuntu@your-server-ip REMOTE_DIR=/opt/model-test ENV_FILE=.env scripts/deploy_remote.sh
```

Use a narrower mode when only config or model files changed:

```bash
scripts/update_service.sh env
scripts/update_service.sh restart
scripts/update_service.sh logs
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh
```

Rebuild the ASR image after vLLM dependency or Dockerfile changes:

```bash
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh build
```
