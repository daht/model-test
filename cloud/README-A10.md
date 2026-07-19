# HY-MT1.5-1.8B on NVIDIA A10

## SenseVoice Small evaluation deployment

The `sensevoice` path loads one local `FunAudioLLM/SenseVoiceSmall` model through
pinned FunASR 1.3.14 on one GPU owner. It uses rolling full-utterance re-decode,
dynamic batches of eight, ITN, a 15-second utterance ceiling, and 2-second
partial updates. These are test settings, not an A10 capacity claim.

On a trusted staging host, select and record an immutable revision, download it
to `models/SenseVoiceSmall`, verify that the official `example/en.mp3` is
present, and create `models/SenseVoiceSmall.manifest.json` with
`python3 -m app.asr_artifacts create`. Transfer the approved directory and
manifest together. Runtime startup is local-only and must not download or
update model files.

Use `cloud/A10.sensevoice.env.example`, keep `API_KEY` supplied only through the
environment, then run `scripts/verify_asr_release.sh release`. R08 verifies the
manifest, pinned Silero asset, model load, and real non-silent streaming warmup.
After deployment, start `scripts/monitor_asr_bottleneck.sh` and run strict
Chinese/Japanese speech at 1/8/16/24/32/64 streams. Record errors, p50/p95
completion overhead, batch fill, GPU utilization, peak VRAM, buffered/reserved
audio, and quality against the current accepted backend. Stop at the first
failed stage; do not alter thresholds after observing it.

Atomic rollback restores the prior matching backend configuration, image,
model directory, and approved manifest as one unit, recreates only
`qwen-asr-api`, then requires `/ready` and strict speech before admission is
reopened.

## faster-whisper large-v3 evaluation deployment

The selectable `faster_whisper` path uses one CTranslate2 `large-v3` model
owner, FP16, rolling partial beam one, utterance-final beam five, and dynamic
cross-session batches of four. It performs transcription only. The initial
`ASR_MAX_ACTIVE_STREAMS=14` in the dedicated example is a test starting point,
not a validated A10 capacity claim.

The candidate uses 200 ms bounded coalescing latency so compatible public
WebSocket streams that arrive slightly offset can share a dynamic batch. Its
15-second maximum utterance boundary caps the repeated full-segment work of the
rolling adapter; longer continuous speech is committed in bounded segments.
The engine suppresses repeated three-token sequences so one pathological item
cannot stall every session in its batch.
Its
`ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS=6.0` setting provides six seconds of jitter headroom, not an accepted lag target. Keep
`ASR_MAX_CONNECTION_LAG_SECONDS=4.0` and
`ASR_MAX_UNDECODED_AGE_SECONDS=8.0` unchanged so sustained lag still fails
explicitly.

On a trusted staging host, choose and record an immutable revision from
`Systran/faster-whisper-large-v3`, then download exactly that revision:

```bash
export APPROVED_FASTER_WHISPER_REVISION='<immutable-hugging-face-commit-sha>'
hf download Systran/faster-whisper-large-v3 \
  --revision "${APPROVED_FASTER_WHISPER_REVISION}" \
  --local-dir models/faster-whisper-large-v3
python3 -m app.asr_artifacts create \
  --model-dir models/faster-whisper-large-v3 \
  --output models/faster-whisper-large-v3.manifest.json \
  --source Systran/faster-whisper-large-v3 \
  --revision "${APPROVED_FASTER_WHISPER_REVISION}"
```

Do not create the approval manifest from an unverified target-host download.
Transfer the model directory and manifest through the authenticated release
channel. Compose intentionally loads the repository root `.env`; `--env-file`
alone does not replace that service-level mapping. Prepare a candidate file and
retain the current Qwen environment and image before the maintenance window:

```bash
cp --preserve=mode .env .env.qwen-rollback
docker tag qwen-asr-api:latest qwen-asr-api:qwen-rollback
cp cloud/A10.faster-whisper.env.example .env.faster-whisper-candidate
chmod 600 .env.faster-whisper-candidate .env.qwen-rollback
editor .env.faster-whisper-candidate
docker compose build qwen-asr-api
```

Stop the current model owner, install the candidate as root `.env`, then run the
backend-aware release gate against that exact environment and approved
artifacts. R08 loads a disposable full model, so the old owner must already be
stopped on a single A10:

```bash
docker compose stop qwen-asr-api
cp .env.faster-whisper-candidate .env
chmod 600 .env
export ASR_RELEASE_ENV_FILE="$PWD/.env"
export ASR_RELEASE_MODEL_DIR="$PWD/models/faster-whisper-large-v3"
export ASR_RELEASE_MANIFEST="$PWD/models/faster-whisper-large-v3.manifest.json"
scripts/verify_asr_release.sh release
docker compose up -d --force-recreate --no-deps --no-build qwen-asr-api
docker compose ps qwen-asr-api
docker compose logs --tail 200 qwen-asr-api
```

Verify `/ready`, strict Chinese/Japanese speech, and a 1/4/8/12/14/16 stream
sweep while recording p50/p95 completion overhead, errors, batch fill, GPU
utilization, peak VRAM, and `session_buffer_high_water_seconds`. Also record
the current buffered and reserved seconds after each stage and require them to
return to zero. Capacity is not accepted until the first failing concurrency
stage is known. Promotion additionally requires an approved
multilingual CER/WER corpus for Chinese, Yue, English, Japanese, and Korean.
The repository's mock tests establish scheduling and protocol behavior only.

Keep the previous Qwen environment and image until the evaluation passes. The
atomic rollback direction is `faster_whisper -> qwen_vllm`: restore the prior
environment with `ASR_BACKEND=qwen_vllm`, `ASR_STREAM_MODE=stateful`, the Qwen
model ID and matching manifest, recreate only `qwen-asr-api`, then require
`/ready` and strict speech verification before reopening admission.

The concrete rollback uses the retained artifacts:

```bash
docker compose stop qwen-asr-api
cp .env.qwen-rollback .env
docker tag qwen-asr-api:qwen-rollback qwen-asr-api:latest
docker compose up -d --force-recreate --no-deps --no-build qwen-asr-api
docker compose ps qwen-asr-api
```

## Semantic Gateway topology

Deploy one ASR process, one Uvicorn worker, and one model owner on each A10.
`docker-compose.yml` starts `app.asr_gateway:app`; the Gateway and selected local
adapter share the ASR runtime image, while CUDA/model execution remains on the
coordinator owner thread. Do not load multiple large ASR models on one GPU by
default.

The stateful `qwen_vllm` runtime pinned to `qwen-asr==0.0.6` and `vllm==0.14.0`
must use `Qwen/Qwen3-ASR-1.7B`, not the `-hf` Transformers export. Credentials enter
through environment variables and must not appear in commands or logs.
ASR WebSocket clients authenticate with `X-API-Key` during the upgrade, fetch
stream-info with the same header, and terminate input with `{"type":"finish"}`.
Body-key authentication and the old `end` command are not supported.

A10 capacity remains unverified until the release and live gates run against
the exact image, model, manifest, configuration, non-silent warmup, speech
corpus, concurrency levels, latency thresholds, and VRAM measurements. Unit
tests and fake dynamic adapters establish scheduling semantics only.

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
hf download Qwen/Qwen3-ASR-1.7B \
  --revision "$APPROVED_QWEN_REVISION" \
  --local-dir models/Qwen3-ASR-1.7B
python3 -m app.asr_artifacts create \
  --model-dir models/Qwen3-ASR-1.7B \
  --output models/Qwen3-ASR-1.7B.manifest.json \
  --source Qwen/Qwen3-ASR-1.7B \
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
ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B.manifest.json
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

## ASR full-chain diagnostic capture

Enable high-frequency structured events only for a diagnostic window:

```bash
sed -i 's/^ASR_DIAGNOSTIC_LOGGING=.*/ASR_DIAGNOSTIC_LOGGING=true/' .env
SERVICE=qwen-asr-api BASE_URL=http://127.0.0.1:8002 scripts/update_service.sh env
```

Start the monitor on the A10 host before starting the client workload:

```bash
set -a
source ./.env
set +a
scripts/monitor_asr_bottleneck.sh
```

The monitor records authenticated Gateway metrics, correlated structured ASR
events, ASR and HY-MT container resources, GPU totals, and GPU processes. Press
Ctrl+C after the workload. The command prints the unique run directory, the
generated `report.md`, and the timestamped archive to provide for analysis.
Credentials, audio, PCM, prompts, and transcript text are forbidden from the
archive. Archive creation fails if the API key is detected.

Retention defaults to 20 completed runs and 14 days. Override it without
changing repository files:

```bash
ASR_MONITOR_KEEP_RUNS=10 ASR_MONITOR_KEEP_DAYS=7 scripts/monitor_asr_bottleneck.sh
```

After the incident window, set `ASR_DIAGNOSTIC_LOGGING=false` and recreate only
the ASR service. Low-frequency terminal, buffer rejection, cleanup conflict,
worker-state, and slow-engine events remain enabled.
