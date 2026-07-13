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

Qwen3-ASR model and approval manifest, created on a trusted staging host:

```bash
cd /opt/model-test
hf download Qwen/Qwen3-ASR-1.7B-hf \
  --revision "$APPROVED_QWEN_REVISION" \
  --local-dir models/Qwen3-ASR-1.7B-hf
python3 -m app.asr_artifacts create \
  --model-dir models/Qwen3-ASR-1.7B-hf \
  --output models/Qwen3-ASR-1.7B-hf.manifest.json \
  --source Qwen/Qwen3-ASR-1.7B-hf \
  --revision "$APPROVED_QWEN_REVISION"
```

`APPROVED_QWEN_REVISION` must come from the release approval process. Transfer
the manifest through the authenticated release channel and do not generate it
from an unverified target-host download. Runtime verification proves that the
target files match that manifest; approval of the revision and manifest is an
external deployment gate.

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

Use the generated value as `API_KEY`. Production Qwen backends reject missing
keys, values shorter than 32 characters, and known placeholders.
Set `DEPLOYED_API_KEY` to the same value when running the client commands below.

For A10, keep:

```bash
TORCH_DTYPE=float16
DEVICE=auto
MAX_NEW_TOKENS=1024
ASR_TORCH_DTYPE=bfloat16
ASR_DEVICE=auto
ASR_REQUIRE_MODEL_MANIFEST=true
ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B-hf.manifest.json
ASR_BACKEND=qwen_vllm
ASR_STREAM_MODE=stateful
ASR_STREAM_CHUNK_SECONDS=1.0
ASR_MAX_UTTERANCE_SECONDS=30.0
ASR_STATE_WATCHDOG_SECONDS=120.0
ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8
ASR_VLLM_MAX_NEW_TOKENS=32
ASR_STREAM_UNFIXED_CHUNK_NUM=2
ASR_STREAM_UNFIXED_TOKEN_NUM=5
ASR_VAD_MODEL_VERSION=6.2.1
ASR_VAD_MODEL_SHA256=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
ASR_VAD_MIN_SPEECH_MS=250
ASR_VAD_MIN_SILENCE_MS=800
ASR_VAD_HANGOVER_MS=160
ASR_VAD_PRE_ROLL_MS=200
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

Do not treat three-service GPU colocation as validated. The Compose file exposes the same A10 to translation, ASR, and TTS, while ASR may reserve the fraction configured by `ASR_VLLM_GPU_MEMORY_UTILIZATION`. Before enabling all three on one 24GB GPU, run the real workload gate below and confirm with `nvidia-smi` that peak allocated memory leaves operational headroom:

1. Start each service alone and record idle and peak GPU memory.
2. Run concurrent translation, WebSocket ASR, and TTS traffic at the intended production concurrency.
3. Reject the topology if any process OOMs, restarts, exceeds its latency SLO, or leaves less than the team's required GPU headroom.
4. If the gate fails, lower the configurable ASR fraction or place services on separate GPUs/instances; do not infer safety from mock tests.

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

Before promotion and again against the deployed service, follow the layered
commit, release, and live gates in `docs/asr-release-verification.md`. The runner
fails closed when Docker, GPU, approved model, manifest, live audio, runtime
secret, or SLO inputs are unavailable.

For an existing ASR deployment, export the external Chinese/Japanese speech
paths and approved overhead/VRAM thresholds, provide `ASR_LIVE_API_KEY` through
the environment or hidden prompt, then run the fail-closed full workflow:

```bash
scripts/deploy_asr_cloud.sh
```

It verifies the release, deploys the exact verified image without rebuilding,
checks local readiness and WebSocket lifecycle, runs the live matrix, and rolls
back on any post-cutover failure. Evidence and rollback backups default to
protected `/secure` directories outside the repository. Use
`scripts/deploy_asr_cloud.sh --dry-run` to inspect the ordered workflow.

This is a planned-downtime deployment on one A10. The wrapper verifies and
receipts the current rollback baseline, stops the old Qwen owner, then runs R08;
it never loads old and candidate models together. After cutover it runs the
receipt-bound `deployed-live` L01-L04 layer, which does not build or start a
second model container. Do not run the full `verify_asr_release.sh live` mode
beside a loaded ASR on the same GPU because full live mode includes R08.

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
API_KEY="$DEPLOYED_API_KEY" BASE_URL=http://your-server-ip:8000 scripts/smoke_test.sh
API_KEY="$DEPLOYED_API_KEY" BASE_URL=http://your-server-ip:8002 scripts/smoke_asr.sh
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

Stateful `partial` text is always replaceable, including punctuation revisions. The pinned Silero VAD v6.2.1 CPU model is the normal endpoint source; it confirms speech before any audio reaches Qwen and finalizes after the configured trailing silence while retaining 200ms pre-roll and 160ms hangover. Continuous speech is normally split at 30 seconds. The independent 120-second watchdog is an invariant failure, not a second rollover policy.

Test WebSocket streaming from the server:

```bash
API_KEY="$DEPLOYED_API_KEY" \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://127.0.0.1:8002/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

Test from your own machine by changing the URL:

```bash
API_KEY="$DEPLOYED_API_KEY" \
python3 scripts/stream_asr_client.py /path/to/audio.wav \
  --url ws://your-server-ip:8002/v1/transcribe/stream \
  --language zh \
  --show-stream-info \
  --print-mode display \
  --realtime
```

Check the deployed ASR mode during smoke tests:

```bash
API_KEY="$DEPLOYED_API_KEY" \
EXPECT_ASR_STREAM_MODE=stateful \
EXPECT_ASR_BACKEND=qwen_vllm \
EXPECT_ASR_STABLE_COMMIT_ENABLED=false \
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
