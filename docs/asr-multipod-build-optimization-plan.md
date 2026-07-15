# ASR Multi-Pod Build Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pinned Qwen/vLLM backend without replacing an older Torch/CUDA stack, reuse downloads after interrupted builds, and run the gateway from a lightweight CPU-only image.

**Architecture:** One shared ASR backend image provides the pinned Torch 2.9.1/CUDA 12.8 runtime and is built by backend 1 only; backend 2 reuses that local image. A separate slim Python image serves only the sticky FastAPI gateway. BuildKit cache mounts preserve downloaded wheels outside final image layers.

**Tech Stack:** Docker BuildKit, Docker Compose, Python 3.11, FastAPI, Uvicorn, HTTPX, websockets, PyYAML, pytest.

---

### Task 1: Lock the image contracts with failing tests

**Files:**
- Modify: `tests/test_asr_deployment_scripts.py`
- Modify: `tests/test_asr_gateway.py`

- [ ] **Step 1: Add the ASR base/cache regression**

Add a test that reads `Dockerfile.asr` and asserts:

```python
assert "FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime" in dockerfile
assert "RUN --mount=type=cache,target=/root/.cache/pip" in dockerfile
assert "PIP_NO_CACHE_DIR" not in dockerfile
```

- [ ] **Step 2: Add the lightweight gateway regression**

Add assertions that `Dockerfile.asr-gateway` and
`requirements-asr-gateway.txt` exist, use `python:3.11-slim`, install the
gateway requirements, and contain none of `torch`, `vllm`, `qwen-asr`, CUDA,
Silero, or model assets.

- [ ] **Step 3: Tighten the Compose topology test**

Assert that backend 1 owns the only ASR build definition, backend 2 uses the
same `qwen-asr-api:latest` image without `build`, backend 2 has
`pull_policy: never`, and the gateway builds `Dockerfile.asr-gateway` as
`qwen-asr-gateway:latest` without GPU or `.env` access.

- [ ] **Step 4: Run the tests and capture RED evidence**

Run:

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_deployment_scripts.py \
  tests/test_asr_gateway.py -q -p no:cacheprovider
```

Expected: failures identify the Torch 2.5.1/CUDA 12.4 base, missing cache
mount, missing gateway image files, and duplicate backend build definitions.

### Task 2: Align the ASR backend image with the pinned runtime

**Files:**
- Modify: `Dockerfile.asr`

- [ ] **Step 1: Verify the exact upstream base tag exists**

Resolve the official `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` tag before
editing. Record a missing tag as a blocker rather than substituting an
unreviewed CUDA or Torch version.

- [ ] **Step 2: Replace the mismatched base and enable reusable pip cache**

Use this structure:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install \
      --index-url "${PIP_INDEX_URL}" \
      --trusted-host "${PIP_TRUSTED_HOST}" \
      -r requirements-asr-vllm.txt
```

Keep the Silero checksum, Qwen streaming contract check, application copy, and
labels unchanged.

- [ ] **Step 3: Run the focused deployment test**

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_deployment_scripts.py -q -p no:cacheprovider
```

Expected: PASS.

### Task 3: Add the lightweight gateway image

**Files:**
- Create: `requirements-asr-gateway.txt`
- Create: `Dockerfile.asr-gateway`
- Test: `tests/test_asr_gateway.py`

- [ ] **Step 1: Add pinned gateway-only requirements**

Create:

```text
fastapi==0.139.0
uvicorn[standard]==0.38.0
httpx==0.28.1
websockets==16.0
```

- [ ] **Step 2: Add the slim gateway Dockerfile**

Use Python 3.11 slim, a BuildKit pip cache mount, copy only the gateway
requirements and `app`, expose port 8000, and start
`app.asr_gateway:app`. Do not install system model/audio packages or copy model
assets.

- [ ] **Step 3: Run the gateway image-contract test**

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py -q -p no:cacheprovider
```

Expected: the lightweight image assertions pass with all existing gateway
protocol tests still green.

### Task 4: Build the backend once and reuse it

**Files:**
- Modify: `docker-compose.asr-multipod.yml`
- Test: `tests/test_asr_gateway.py`

- [ ] **Step 1: Remove `build` from the shared backend anchor**

Keep `image: qwen-asr-api:latest` and all runtime settings in the anchor.
Add the ASR build definition only to `qwen-asr-backend-1`.

- [ ] **Step 2: Make backend 2 local-image-only**

Set `pull_policy: never`, leave out `build`, and depend on backend 1 becoming
healthy so model startup is serialized.

- [ ] **Step 3: Point the gateway at its slim image**

Set:

```yaml
build:
  context: .
  dockerfile: Dockerfile.asr-gateway
image: qwen-asr-gateway:latest
```

Keep its CPU-only, credential-free service contract.

- [ ] **Step 4: Run focused Compose and gateway tests**

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py \
  tests/test_asr_deployment_scripts.py -q -p no:cacheprovider
```

Expected: PASS.

### Task 5: Update the operator workflow

**Files:**
- Modify: `docs/asr-multipod-gateway.md`

- [ ] **Step 1: Replace the all-service build command**

Document:

```bash
COMPOSE='docker compose --env-file .env -f docker-compose.asr-multipod.yml'
$COMPOSE build qwen-asr-backend-1 asr-gateway
$COMPOSE up -d --no-build
```

- [ ] **Step 2: Add version and GPU smoke checks**

Document commands that print Torch, Torch CUDA, vLLM, Transformers, GPU name,
and `/ready` for backend 1 before backend 2 is started. Credentials remain in
the environment and never appear in commands or logs.

- [ ] **Step 3: Preserve the strict capacity ladder**

Keep the deterministic client start gate and require both backend readiness,
VRAM evidence, latency, failures, and transcript accuracy before claiming a
capacity improvement.

### Task 6: Verify, stage, and commit the candidate

**Files:**
- All intended files from Tasks 1-5

- [ ] **Step 1: Run focused regression**

```bash
/model/.venv/bin/python -m pytest \
  tests/test_asr_gateway.py \
  tests/test_asr_deployment_scripts.py \
  tests/test_asr_config.py \
  tests/test_verify_asr_release.py -q -p no:cacheprovider
```

- [ ] **Step 2: Stage only intended paths**

```bash
git add -- \
  Dockerfile.asr \
  Dockerfile.asr-gateway \
  requirements-asr-gateway.txt \
  docker-compose.asr-multipod.yml \
  docs/asr-multipod-gateway.md \
  tests/test_asr_deployment_scripts.py \
  tests/test_asr_gateway.py
```

- [ ] **Step 3: Run the staged commit gate**

```bash
PATH=/model/.venv/bin:$PATH scripts/verify_asr_release.sh commit
```

Expected: every commit gate passes. Docker build, real A10 warmup, dual-model
VRAM, and live capacity remain explicit test-server gaps.

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(asr): optimize multi-pod image builds"
```
