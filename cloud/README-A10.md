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
ASR_STREAM_CHUNK_SECONDS=4.0
```

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
```

From your local machine:

```bash
REMOTE_HOST=root@your-server-ip ENV_FILE=.env scripts/deploy_remote.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8002 scripts/smoke_asr.sh
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

Test WebSocket streaming from the server:

```bash
API_KEY=your-api-key \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh \
  --realtime
```

Test from your own machine by changing the URL:

```bash
API_KEY=your-api-key \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://your-server-ip:8002/v1/transcribe/stream \
  --language zh \
  --realtime
```

## 6. Operations

```bash
docker compose ps
docker compose logs -f hy-mt-api
docker compose logs -f qwen-asr-api
docker stats
nvidia-smi
docker compose restart hy-mt-api
docker compose restart qwen-asr-api
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
