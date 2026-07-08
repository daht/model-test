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
```

From your local machine:

```bash
REMOTE_HOST=root@your-server-ip ENV_FILE=.env scripts/deploy_remote.sh
API_KEY=your-api-key BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
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

## 6. Operations

```bash
docker compose ps
docker compose logs -f hy-mt-api
docker stats
nvidia-smi
docker compose restart hy-mt-api
docker compose pull && docker compose up --build -d
```

## 7. Updates

Run this from the project directory on the server:

```bash
cd /opt/model-test
scripts/update_service.sh
```

Use a narrower mode when only config or model files changed:

```bash
scripts/update_service.sh env
scripts/update_service.sh restart
scripts/update_service.sh logs
```
