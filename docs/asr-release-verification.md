# ASR Release Verification

Use `scripts/verify_asr_release.sh` as the single entry point for ASR change,
release, and deployed-service verification. It is fail closed: a selected mode
returns success only when every gate in that mode and every lower layer passes.
It never treats a missing tool, model, GPU, URL, audio file, threshold, or secret
as a skipped gate.

The runner does not modify tracked files, the index, commits, or deployed
configuration. Temporary bytecode, rendered Compose configuration, client logs,
and GPU samples are created in a mode-700 directory under `${TMPDIR:-/tmp}` and
removed on exit. Do not run independent or product acceptance through this
script; those remain separate approvals after engineering verification passes.

## Commands and ownership

Inspect the interface or an ordered plan without executing gates:

```bash
scripts/verify_asr_release.sh --help
scripts/verify_asr_release.sh --list-gates live
scripts/verify_asr_release.sh --dry-run release
```

### Commit mode

The developer runs commit mode before each commit and before requesting review.
A fresh checkout is not required. Stage every intended new file first so the
index secret, forbidden-path, binary, and size gates cover it.

```bash
git add -- <intended-files>
scripts/verify_asr_release.sh commit
```

Commit mode runs the full explicit-mock pytest suite exactly once, with an
ephemeral generated credential supplied only to the mock process. It also redirects
`compileall` bytecode under temporary storage, checks every tracked shell script
with `bash -n`, checks staged and unstaged whitespace, scans tracked worktree and
index content for high-confidence credentials, rejects forbidden tracked paths,
and rejects binary or files larger than 1 MiB in the staged delta. Override the
size limit only through a reviewed invocation of
`ASR_VERIFY_MAX_STAGED_BYTES`; increasing it does not allow forbidden artifacts.

### Release mode

The release engineer or release CI runs release mode before image promotion.
Release mode includes commit mode and requires a fresh, clean checkout or clean
CI workspace. Ignored operator assets may be present, but `git status
--porcelain=v1 --untracked-files=all` must otherwise be empty.

Create the production `.env` at the repository root without tracking it. The
Compose file reads that exact path. Put the approved model and its separately
delivered manifest below the ignored repository `models/` directory, then set:

```bash
export ASR_RELEASE_ENV_FILE="$PWD/.env"
export ASR_RELEASE_MODEL_DIR="$PWD/models/Qwen3-ASR-1.7B-hf"
export ASR_RELEASE_MANIFEST="$PWD/models/Qwen3-ASR-1.7B-hf.manifest.json"
scripts/verify_asr_release.sh release
```

The `.env` must configure `qwen_vllm`, stateful streaming, required model
manifest verification, eager loading, disabled file transcription, one Uvicorn
worker, a read-only `/models` mount, and a non-placeholder production API key.
The configured container paths must match the two host asset paths above.

Release mode validates the exact model set against the operator-approved
manifest, renders and checks Docker Compose configuration, builds
`qwen-asr-api:latest`, reruns the pinned Qwen/vLLM contract and Silero checksum
inside that image, checks host and container GPU access, and starts a disposable
Compose container for real Qwen model, manifest, VAD, and streaming warmup. The
container uses `--rm`; a failure in Docker build/runtime, GPU access, Qwen load,
Silero verification, or warmup fails the release.

### Live mode

The release engineer or service operator runs live mode only immediately before
rollout against the release candidate deployment, or immediately after a
deployment against the deployed service. It includes release and commit mode,
so use the same fresh validation checkout, `.env`, approved model, Docker host,
and GPU host described above.

Supply two external files containing actual speech. Do not copy or generate
audio under the repository. Provide the deployed API key only through the
environment; never place it in a command argument, shell history, repository
file, evidence filename, or issue text.

```bash
export ASR_LIVE_BASE_URL="https://asr.example.internal"
export ASR_LIVE_WS_URL="wss://asr.example.internal/v1/transcribe/stream"
export ASR_LIVE_ZH_AUDIO="/secure/release-input/chinese-speech.flac"
export ASR_LIVE_JA_AUDIO="/secure/release-input/japanese-speech.flac"
export ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS="10"
export ASR_LIVE_MAX_GPU_MEMORY_MIB="23000"
export ASR_LIVE_GPU_INDEX="0"
read -rsp "Deployed ASR API key: " ASR_LIVE_API_KEY
echo
export ASR_LIVE_API_KEY
scripts/verify_asr_release.sh live
unset ASR_LIVE_API_KEY
```

Choose the overhead and VRAM limits from the approved service SLO and GPU
headroom policy; the example values are inputs, not performance claims. Live
mode runs the existing readiness/WebSocket smoke, then Chinese and Japanese
speech at 200 ms and 500 ms real-time chunks. Strict client validation requires
`ready` at sequence 1, continuous event sequences, no `error`, at least one
`sentence_final`, exactly one terminal `final`, no later event, and normal close
code 1000. It then runs the
Chinese and Japanese 200 ms streams concurrently and enforces completion overhead
for every stream. GPU `memory.used` is polled every 0.25 seconds throughout the
live gates. L04 requires zero failed `nvidia-smi` queries, at least four valid
samples, at least 0.75 seconds between the first and last valid samples, and a
sampled maximum no greater than `ASR_LIVE_MAX_GPU_MEMORY_MIB`. Attempt, success,
failure, and monotonic timing data remain in the protected temporary directory
until the runner removes it on exit.

The two-stream gate matches the conservative default rollout concurrency. A
higher intended concurrency or multi-service GPU topology still requires its
separately approved capacity workload; do not infer 4- or 8-stream capacity,
first-partial p95, or multi-service headroom from this gate.

## Gate inventory

| Gate | Pass criterion |
| --- | --- |
| C01 | Full repository pytest suite passes once with all three backends explicitly mocked. |
| C02 | All repository Python sources compile with output redirected under temporary storage. |
| C03 | Every tracked shell script passes `bash -n`. |
| C04 | Both staged and unstaged `git diff --check` pass. |
| C05 | No recognized high-confidence credential occurs in tracked worktree or index content. |
| C06 | No forbidden tracked path, binary staged delta, or staged file above the size limit exists. |
| R01 | Checkout is clean before and after the commit gates. |
| R02 | Docker, Compose, daemon, NVIDIA GPU, `.env`, model, and manifest are available. |
| R03 | Rendered ASR Compose configuration matches the production stateful contract. |
| R04 | Every host model artifact, size, and SHA256 matches the approved manifest. |
| R05 | The ASR image builds with pinned runtime and Silero build checks. |
| R06 | Runtime contract and Silero checksum pass inside the built image. |
| R07 | Both host and disposable image can access the selected NVIDIA runtime. |
| R08 | Disposable real-Qwen startup completes manifest, VAD, load, and streaming warmup. |
| L01 | Deployed health, readiness, stream-info, and synthetic WebSocket lifecycle pass. |
| L02 | All four zh/ja and 200/500 ms strict speech cases pass. |
| L03 | Concurrent zh and ja streams both pass. |
| L04 | Stream overhead passes; GPU sampling has zero failures, at least four valid samples spanning 0.75 seconds, and stays within the supplied limit. |

## Pass, evidence, and failure handling

The mode passes only when the runner prints `ASR <mode> verification passed.`
Any nonzero exit is a failed verification. Missing prerequisites are listed
together before expensive lower-layer work begins; there is no successful skip.

Retain the commit SHA, runner mode, UTC start time, host/image identity, approved
manifest revision, threshold inputs, complete terminal output, and exit status
in the release evidence system outside this repository. To capture terminal
output without losing the runner status:

```bash
set -o pipefail
scripts/verify_asr_release.sh release 2>&1 \
  | tee "/secure/release-evidence/asr-release-$(git rev-parse HEAD).log"
```

The runner does not print the runtime secret. Evidence access must still be
restricted because live output contains service topology and transcripts.

On failure, stop promotion or rollout, retain the failed evidence, correct the
earliest failing gate, and rerun the entire selected mode. Do not bypass a gate,
raise a threshold after observing a failure without approval, generate a new
manifest from suspect target files, or combine evidence from partial runs.

The rollback rule is atomic: if live mode fails after deployment, restore the
last accepted image digest, configuration, model directory, and matching
approved manifest together. Confirm readiness, then run live mode against the
restored deployment. Do not roll back only code while leaving an incompatible
model, VAD asset, config, or manifest in place.
