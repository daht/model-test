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
export ASR_RELEASE_MODEL_DIR="$PWD/models/Qwen3-ASR-1.7B"
export ASR_RELEASE_MANIFEST="$PWD/models/Qwen3-ASR-1.7B.manifest.json"
scripts/verify_asr_release.sh release
```

For faster-whisper, use the matching environment and artifact paths instead:

```bash
cp cloud/A10.faster-whisper.env.example .env
chmod 600 .env
editor .env
export ASR_RELEASE_ENV_FILE="$PWD/.env"
export ASR_RELEASE_MODEL_DIR="$PWD/models/faster-whisper-large-v3"
export ASR_RELEASE_MANIFEST="$PWD/models/faster-whisper-large-v3.manifest.json"
scripts/verify_asr_release.sh release
```

Back up the previous root `.env` before replacing it. The runner intentionally
requires `ASR_RELEASE_ENV_FILE` to be the repository root `.env` because that is
the service-level Compose `env_file` mapping.

The environment must configure either `qwen_vllm` with `stateful` streaming or
`faster_whisper` with `rolling` streaming. The faster-whisper release contract
additionally fixes FP16, batch four, partial beam one, final beam five, and
`transcribe`. Both paths require model manifest verification, eager loading,
disabled file transcription, one Uvicorn worker, a read-only `/models` mount,
and a non-placeholder production API key. The configured container paths must
match the two host asset paths above.

Release mode validates the exact model set against the operator-approved
manifest, renders and checks Docker Compose configuration, builds
`qwen-asr-api:latest`, reruns the pinned Qwen/vLLM and faster-whisper package
contracts plus the Silero checksum inside that image, checks host and container
GPU access, and starts a disposable Compose container for the selected real
model, manifest, VAD, and backend warmup. The container uses `--rm`; a failure
in Docker build/runtime, GPU access, model load, Silero verification, or warmup
fails the release.

### One-command cloud deployment

On an existing cloud ASR host, `scripts/deploy_asr_cloud.sh` orchestrates the
release and deployed-live runners without weakening their gates. It requires a clean
committed checkout, the repository `.env`, an approved model and matching
manifest under `models/`, and exactly one currently deployed ASR container to
serve as the rollback point. It prepares the ignored repository `.venv` from
the pinned `requirements-dev.txt` only when needed; it never creates a manifest
or changes Git state.

Set the live audio and approved thresholds shown below, provide
`ASR_LIVE_API_KEY` through the environment or the wrapper's hidden prompt, then
run:

```bash
scripts/deploy_asr_cloud.sh
```

Full release, exact-image cutover, local readiness/WebSocket smoke, and
receipt-bound deployed-live verification are the default. Protected evidence and rollback backups default
to `/secure/asr-release-evidence` and `/secure/asr-release-backup`; override
them with `ASR_DEPLOY_EVIDENCE_DIR` and `ASR_DEPLOY_BACKUP_DIR`, which must stay
outside the repository. Before maintenance it compares the running container's
effective environment, command, read-only model mount, and start time with the
current Compose configuration and artifact timestamps. It revalidates the
approved manifest and requires current readiness/WebSocket smoke, then records
a protected rollback-baseline receipt. Prechanged or unmatched config, model,
manifest, image, or Compose state fails before the service is stopped.

The wrapper then stops the existing ASR container and confirms that no model
owner remains. This begins a planned maintenance window that includes the full
release build and R08 warmup. Only after R08 exits does the wrapper deploy the
release-verified image ID with Compose `--no-build`. It records a release
receipt bound to the clean commit SHA, exact image ID, `.env`, manifest, model
path, and protected release evidence. It does not modify model files.

Any error or HUP/INT/TERM after maintenance starts attempts to stop the current
owner and restore the receipted previous image,
`.env`, and approved manifest, revalidates the unchanged model directory, and
revalidates the recreated container against the rollback baseline, then runs
restored readiness and smoke. The original failing status is preserved;
rollback failure is reported separately. Use `--dry-run` to inspect the plan
without prerequisites. `--skip-live` must be deliberate and does not produce
live-verified evidence.

### Live mode

The release engineer or service operator runs live mode only immediately before
rollout against the release candidate deployment, or immediately after a
deployment against the deployed service. It includes release and commit mode,
so use the same fresh validation checkout, `.env`, approved model, Docker host,
and GPU host described above.

`live` includes R08 and therefore starts a disposable full selected ASR model. Do not
run it while another model-owning ASR container is loaded on the same GPU. On a
single A10, use the maintenance workflow above. Its post-cutover command is the
receipt-bound layer:

```bash
export ASR_DEPLOYED_RELEASE_RECEIPT=/secure/asr-release-backup/<release-id>/release.receipt.json
scripts/verify_asr_release.sh deployed-live
```

`deployed-live` first proves D01: the clean candidate SHA, exact running image
ID, `.env`, model path, manifest hashes, protected release evidence, and current
Compose production contract must match the release receipt. It then runs the
same L01-L04 smoke, speech matrix, concurrency, overhead, and VRAM gates. It
never builds an image, invokes Compose `run`, warms a model, or recreates the
service. Without the matching protected receipt it fails closed and cannot be
used as release evidence.

Supply two external files containing actual speech. Do not copy or generate
audio under the repository. Provide the deployed API key only through the
environment; never place it in a command argument, shell history, repository
file, evidence filename, or issue text. The streaming client's strict
`--verify-protocol` mode accepts the key only from the `API_KEY` environment
variable and rejects `--api-key`, even when both are present. The argument
remains available for ordinary, non-strict manual use. This source distinction
is explicit; the client never compares or prints key values to infer where a
key came from.

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
code 1000 received from the server. Malformed JSON fails immediately. A locally
sent close frame is not proof of server closure, and a missing received close
frame fails strict verification. It then runs the
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
| R03 | Rendered ASR Compose configuration matches the selected backend contract. |
| R04 | Every host model artifact, size, and SHA256 matches the approved manifest. |
| R05 | The ASR image builds with pinned runtime and Silero build checks. |
| R06 | Runtime contract and Silero checksum pass inside the built image. |
| R07 | Both host and disposable image can access the selected NVIDIA runtime. |
| R08 | Disposable selected-ASR startup completes manifest, VAD, load, and backend warmup. |
| L01 | Deployed health, readiness, stream-info, and synthetic WebSocket lifecycle pass. |
| L02 | All four zh/ja and 200/500 ms strict speech cases pass. |
| L03 | Concurrent zh and ja streams both pass. |
| L04 | Stream overhead passes; GPU sampling has zero failures, at least four valid samples spanning 0.75 seconds, and stays within the supplied limit. |
| D01 | A protected release receipt matches the clean SHA, exact running image ID, current config/model/manifest, and successful release evidence before deployed-only L01-L04. |

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

## Diagnostic evidence is supplemental

`scripts/monitor_asr_bottleneck.sh` produces timestamped ASR diagnostic runs and
invokes `scripts/analyze_asr_bottleneck.py` at finalization. This evidence can
explain scheduler fragmentation, real faster-whisper engine grouping, inference
tails, buffer rejection sources, cleanup ownership, and HY-MT GPU contention.
It does not replace commit, release, live, or deployed-live gates.

Diagnostic evidence remains outside the repository. Never stage its run
directories, archives, logs, reports, audio, or credentials. Enable
`ASR_DIAGNOSTIC_LOGGING=true` only for the observation window and recreate the
ASR service after changing the setting.
