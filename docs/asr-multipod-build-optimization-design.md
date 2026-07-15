# ASR Multi-Pod Build Optimization Design

## Objective

Make the experimental multi-Pod deployment reproducible and materially faster
to build without changing the approved Qwen runtime, model, streaming protocol,
or capacity settings.

## Runtime contract

The ASR image must preserve these exact runtime dependencies:

- `qwen-asr[vllm]==0.0.6`
- `vllm==0.14.0`
- `torch==2.9.1`
- `transformers==4.57.6`
- `accelerate==1.12.0`
- Qwen toolkit checkpoint `Qwen/Qwen3-ASR-1.7B`

Because vLLM 0.14.0 requires Torch 2.9.1 and the Linux Torch wheel uses CUDA
12.8 libraries, the ASR Docker base must provide the matching Torch 2.9.1 /
CUDA 12.8 runtime. The build must not start from Torch 2.5.1 / CUDA 12.4 and
then replace the complete framework and CUDA userspace stack with pip wheels.

## Image boundaries

### ASR backend image

`Dockerfile.asr` remains the model-serving image. It contains the Qwen runtime,
vLLM, Torch, Silero VAD asset, ffmpeg, and application code. Both ASR backend
containers use the same built image.

The dependency installation layer uses a BuildKit pip cache mount. Downloaded
wheels remain available to a later build after an interrupted or failed install,
while the final runtime image does not retain the pip cache.

### Gateway image

A separate lightweight gateway image contains only Python, FastAPI, Uvicorn,
HTTPX, websockets, and `app/asr_gateway.py`. It must not install Torch, vLLM,
qwen-asr, model assets, CUDA libraries, or receive GPU access.

## Compose topology

The Compose topology keeps three runtime services:

- `qwen-asr-backend-1`: builds and tags `qwen-asr-api:latest`.
- `qwen-asr-backend-2`: reuses `qwen-asr-api:latest` without a second build
  definition.
- `asr-gateway`: builds the lightweight gateway image and exposes host port
  `8002`.

The documented full startup path builds the backend image once, builds the
gateway image once, and then starts all services with `--no-build`. Directly
starting backend 2 without first building the shared backend image is outside
the supported operator workflow and must fail visibly rather than pull an
unapproved remote image.

## Failure handling

- A missing or incompatible PyTorch base tag fails during image resolution.
- Runtime package contract checks continue to run inside the ASR image build.
- The gateway readiness contract remains fail-closed until both backends are
  ready.
- Real GPU compatibility, model warmup, dual-model VRAM headroom, and capacity
  remain release/live gates and cannot be established by repository tests.

## Verification

Deterministic repository tests must prove:

1. The ASR base names Torch 2.9.1 and CUDA 12.8.
2. The ASR dependency layer uses a BuildKit pip cache mount.
3. The gateway image uses a slim Python base and gateway-only requirements.
4. Only backend 1 owns the ASR build definition; backend 2 reuses the same
   local image and never requests a GPU build of its own.
5. The gateway image and Compose service contain no ASR runtime dependency or
   GPU assignment.
6. The default single-owner `docker-compose.yml` remains unchanged.

The candidate must pass focused gateway/deployment tests and
`scripts/verify_asr_release.sh commit`. Image build, A10 warmup, and strict live
capacity tests remain explicit external gates for the test server.
