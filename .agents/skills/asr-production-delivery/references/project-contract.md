# Project ASR Contract

## Canonical sources

Read only the sources needed for the request:

- `README.md`: deployed architecture and supported operating contract.
- `docs/asr-hardening-beginner-guide.md`: protocol and state-machine explanation.
- `docs/asr-release-verification.md`: authoritative gate ownership and evidence rules.
- `scripts/verify_asr_release.sh`: single executable verification entry point.
- `scripts/stream_asr_client.py`: manual and strict WebSocket client.
- `app/asr.py`: ASR backends and stateful session ownership.
- `app/asr_api.py`: HTTP/WebSocket lifecycle, endpointing, lag, and protocol events.
- `app/asr_inference.py`: single-owner inference coordinator and cleanup ordering.
- `app/asr_streaming.py`: confirmed-prefix and replaceable-tail state machine.
- `app/asr_vad.py`: Silero VAD framing, pre-roll, onset, hangover, and endpointing.
- `app/asr_artifacts.py`: approved model manifest verification.
- `app/config.py`: configuration validation and production defaults.
- `requirements-asr-vllm.txt`: pinned Qwen ASR, vLLM, and Transformers-compatible runtime contract.

Use `rg` to locate current definitions before relying on remembered line numbers.

## Required invariants

Every relevant change must preserve or explicitly test these invariants:

1. PCM samples have one owner and are neither lost nor duplicated across VAD,
   onset retries, endpoint flush, utterance rollover, or `finish_input()`.
2. `partial` replaces the current tail. `sentence_final` permanently appends a
   confirmed segment. `final` contains only the final unconfirmed tail.
3. Event sequences are continuous. Exactly one `final` is terminal; no event may
   follow it. Strict success requires a server-received close frame with code 1000.
4. Confirmed text never changes. A conflict fails the session instead of
   duplicating or rewriting confirmed output.
5. A VAD endpoint, explicit segment, maximum utterance boundary, or end-of-input
   may commit text. Model punctuation alone is replaceable in stateful mode.
6. Pre-roll, candidate, active speech, hangover, remainder, and discarded samples
   obey sample conservation at every state transition.
7. Slow fully-decoded calls, queued audio, undecoded age, and connection lag are
   bounded and fail explicitly rather than accumulate indefinitely.
8. `ASR_MAX_UTTERANCE_SECONDS` is an exact maximum. A frame crossing the boundary
   is split, and the remainder continues in a new official model state.
9. Successful coordinator results are published only after cleanup, reservation
   release, and queue accounting complete. Cleanup failure clears readiness and
   fails the affected job.
10. One ASR process, one Uvicorn worker, and one model owner serve one GPU. File
    transcription remains disabled on the live stateful service.
11. Readiness requires the approved model manifest, pinned Silero asset, model
    load, and real non-silent streaming warmup.
12. Production and strict verification credentials enter through environment
    variables only and are never printed.
13. The pinned `qwen-asr==0.0.6` / `vllm==0.14.0` stateful runtime uses the
    Qwen toolkit checkpoint `Qwen/Qwen3-ASR-1.7B`, not the native Transformers
    export `Qwen/Qwen3-ASR-1.7B-hf`. Never carry a model ID from the `qwen`
    backend into `qwen_vllm` without revalidating the official runtime/model
    contract and real R08 warmup.

## Pinned model/runtime compatibility

For `ASR_BACKEND=qwen_vllm` and `ASR_STREAM_MODE=stateful`:

- use `Qwen/Qwen3-ASR-1.7B` (or its matching local directory and approved
  manifest) with the repository-pinned `qwen-asr==0.0.6`, `vllm==0.14.0`, and
  `transformers==4.57.6` dependency chain;
- do not use `Qwen/Qwen3-ASR-1.7B-hf`; that native Transformers export targets
  a different processor contract and was observed to load as
  `Qwen2TokenizerFast` without `audio_token` under the pinned runtime;
- treat `AttributeError: Qwen2TokenizerFast has no attribute audio_token` as a
  model/runtime mismatch first, not as a GPU-memory, VAD, or streaming-state
  failure;
- remember that manifest verification proves the selected files match the
  recorded hashes. It does not prove that the model export is compatible with
  the selected backend or pinned Python packages. R08 real model warmup remains
  mandatory for a production claim.

This constraint records the July 2026 migration failure where the original
Transformers `qwen` backend selected the `-hf` export and the later
`qwen_vllm` backend inherited that model ID without a compatibility review.

## Repository ownership

- Record the initial Git status and preserve it verbatim outside intended paths.
- Do not use destructive Git commands or broad staging.
- Stage explicit intended files so commit gates inspect the index.
- Do not add audio, model weights, `.env`, generated evidence, HTML reports,
  caches, bytecode, or any `superpowers/` path.
- A Development Agent may edit and commit. A Test Agent and product acceptance
  remain read-only.

## Evidence language

Use these states precisely:

- `implemented`: code exists; verification may still be incomplete.
- `commit-verified`: local/mock repository gates passed.
- `release-verified`: real image, GPU, model, manifest, and warmup gates passed.
- `live-verified`: deployed strict speech, concurrency, latency, and VRAM gates passed.
- `production-ready`: all gates required by the requested rollout passed against
  the same accepted code, image, model, manifest, and configuration.
