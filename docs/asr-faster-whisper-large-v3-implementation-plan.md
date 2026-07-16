# faster-whisper large-v3 Backend Implementation Plan

## Execution Contract

**Goal:** Deploy a selectable, batched `large-v3` streaming backend without
removing the Qwen rollback path.

**Architecture:** Reuse the semantic Gateway scheduler and add a rolling
`FasterWhisperAdapter` backed by a pinned CTranslate2 engine. Endpoint jobs are
distinguished in the scheduler batch key so partial beam-one and final beam-five
work never mix.

**Non-goals:** Model quantization, translation, word timestamps, changing the
public protocol, and claiming A10 production capacity from mock tests.

**Risk:** The engine uses the pinned faster-whisper 1.2.1 batched generation
contract. A package upgrade requires its own adapter compatibility test and A10
gate.

## File Map

- `app/asr_faster_whisper.py`: CTranslate2 engine wrapper and rolling adapter.
- `app/config.py`: backend, rolling mode, compute, batch, beam, and task settings.
- `app/asr_gateway.py`: runtime selection and final-job batch identity.
- `requirements-asr-faster-whisper.txt`: pinned runtime dependencies.
- `Dockerfile.asr`: install the additional selectable runtime.
- `.env.example`, `cloud/A10.env.example`: exact deployment knobs.
- `README.md`, `cloud/README-A10.md`: model staging, deployment, verification,
  and rollback operations.
- `tests/test_asr_faster_whisper.py`: adapter and engine-boundary behavior.
- `tests/test_asr_config.py`, `tests/test_asr_gateway.py`: selection and routing.

## TDD Tasks

1. Add failing configuration tests that accept only
   `faster_whisper+rolling+transcribe`, validate batch/beam settings, require a
   production API key, and preserve all Qwen pairs. Run the focused test and
   confirm it fails because the backend literal and settings do not exist.
2. Add failing adapter tests with an injected recording engine. Prove two
   sessions become one engine batch, buffers remain isolated, partial uses beam
   one, endpoint final uses beam five, auto language locks, mixed locked
   languages sub-batch, results retain input order, and finish/abort clear state.
3. Implement the minimal engine protocol, session state, capabilities, PCM
   conversion, batching, language grouping, cached final result, lifecycle, and
   sanitized error handling needed for those tests.
4. Add failing Gateway tests proving endpoint and finish jobs use a final batch
   key and that `_default_runtime` selects the faster adapter while Qwen still
   selects `LocalCoordinatorAdapter`.
5. Implement rolling stream-mode routing, endpoint final marking, decoding
   identity separation, manifest verification before engine load, and adapter
   selection.
6. Pin `faster-whisper==1.2.1` and `ctranslate2==4.8.1`, install them in the ASR
   image, and add deployment contract tests for the Docker and environment
   wiring.
7. Document trusted model download and manifest creation, A10 startup, strict
   client validation, capacity/VRAM capture, and the atomic environment rollback
   to `qwen_vllm+stateful`.
8. Run focused ASR tests, then the complete suite with mock translation/ASR.
   Explicitly stage only intended files and run
   `scripts/verify_asr_release.sh commit` before committing.

## Gates And Checkpoints

At 30 minutes, require a failing test plus a supported adapter boundary. At 60
minutes, require a testable candidate or split documentation from runtime code.
Local completion is `commit-verified` only. `release-verified` and
`live-verified` require the external A10 model, manifest, Docker, GPU, speech,
latency, accuracy, VRAM, and rollback evidence described in
`docs/asr-release-verification.md`.
