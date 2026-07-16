#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN=0
LIST_GATES=0
MODE=""
PYTHON_BIN=""
TEMP_DIR=""
GPU_MONITOR_PID=""
GPU_MONITOR_STATE=""
MISSING=()

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_asr_release.sh [--dry-run] [--list-gates] commit|release|live|deployed-live
  scripts/verify_asr_release.sh --list-gates [commit|release|live|deployed-live]

Modes:
  commit   Repeatable local gate; a fresh checkout is not required.
  release  Commit gates plus clean checkout, Docker, GPU, config, artifacts, and warmup.
  live     Release gates plus deployed smoke, speech matrix, concurrency, latency, and VRAM.
  deployed-live
            Evidence-bound deployed gates only; does not build, warm up, or start another model.

Options:
  --dry-run     Print the ordered command plan without checking prerequisites or executing gates.
  --list-gates  Print gate identifiers and pass criteria without executing gates.
  -h, --help    Show this help.

Release environment:
  ASR_RELEASE_ENV_FILE       Must identify the repository .env file. Default: .env
  ASR_RELEASE_MODEL_DIR      Host path to the approved ASR model directory.
  ASR_RELEASE_MANIFEST       Host path to the operator-approved model manifest.

Deployed-live environment:
  ASR_DEPLOYED_RELEASE_RECEIPT  Protected receipt created after a successful release gate.

Live environment:
  ASR_LIVE_BASE_URL                    Deployed HTTP base URL.
  ASR_LIVE_WS_URL                      Deployed WebSocket endpoint.
  ASR_LIVE_API_KEY                     Runtime-only secret; never printed or passed as an argument.
  ASR_LIVE_ZH_AUDIO                    External Chinese speech audio path.
  ASR_LIVE_JA_AUDIO                    External Japanese speech audio path.
  ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS Maximum completion overhead above audio duration.
  ASR_LIVE_MAX_GPU_MEMORY_MIB          Maximum sampled memory.used for ASR_LIVE_GPU_INDEX.
  ASR_LIVE_GPU_INDEX                   GPU index to sample. Default: 0
EOF
}

list_commit_gates() {
  cat <<'EOF'
C01 full explicit-mock pytest: all repository tests pass once in a fresh process
C02 compileall: app, tests, and scripts compile with bytecode redirected under a temporary directory
C03 shell syntax: bash -n passes for every tracked shell script
C04 whitespace: staged and unstaged git diff checks pass
C05 high-confidence secret scan: tracked worktree and index contain no recognized credential material
C06 repository artifacts: no forbidden tracked paths and no binary or large staged delta
EOF
}

list_release_gates() {
  cat <<'EOF'
R01 release checkout: commit gates run from a clean checkout or clean CI workspace
R02 release prerequisites: Docker Compose, Docker daemon, NVIDIA runtime, .env, model, and manifest exist
R03 Compose config: rendered qwen-asr-api config matches its selected production backend contract
R04 model provenance: exact host artifact set matches the operator-approved manifest
R05 ASR image build: Dockerfile.asr completes its pinned dependency and Silero checksum build gates
R06 image runtime: pinned Qwen and faster-whisper contracts plus Silero checksum pass inside the image
R07 container GPU runtime: host and built image can access the NVIDIA GPU
R08 real ASR warmup: one-shot container loads the selected approved model, VAD, and decode path
EOF
}

list_live_gates() {
  cat <<'EOF'
L01 deployed smoke: health, readiness, stream-info contract, and synthetic WebSocket lifecycle pass
L02 speech matrix: zh and ja audio pass strict protocol validation at 200 ms and 500 ms real-time chunks
L03 concurrent speech: zh and ja 200 ms streams pass together with no protocol or server error
L04 live SLO: overhead passes; GPU has zero query failures and at least 4 samples spanning 0.75s
EOF
}

list_deployed_binding_gate() {
  cat <<'EOF'
D01 deployed release binding: clean SHA, config, model, manifest, running image ID, and protected release evidence match one receipt
EOF
}

list_deployed_live_gates() {
  list_deployed_binding_gate
  list_live_gates
}

list_gates() {
  case "${MODE:-all}" in
    commit)
      list_commit_gates
      ;;
    release)
      list_commit_gates
      list_release_gates
      ;;
    live)
      list_commit_gates
      list_release_gates
      list_live_gates
      ;;
    all)
      list_commit_gates
      list_release_gates
      list_live_gates
      list_deployed_binding_gate
      ;;
    deployed-live)
      list_deployed_live_gates
      ;;
  esac
}

print_commit_plan() {
  cat <<'EOF'
C04 git diff --check
C04 git diff --cached --check
C05 high-confidence secret scan of tracked worktree and index
C06 forbidden tracked paths plus binary/large staged delta checks
C03 git ls-files '*.sh' | bash -n each script
C02 PYTHONPYCACHEPREFIX=<temporary>/pycache python -m compileall -q app tests scripts
C01 MODEL_BACKEND=mock ASR_BACKEND=mock ASR_STREAM_MODE=chunked ASR_REQUIRE_MODEL_MANIFEST=false ASR_MODEL_MANIFEST_PATH=<empty> ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8 ASR_VLLM_MAX_MODEL_LEN=65536 TTS_BACKEND=mock API_KEY=<ephemeral> python -m pytest tests -q
EOF
}

print_release_plan() {
  print_commit_plan
  cat <<'EOF'
R01 require clean git checkout
R02 require docker compose, Docker daemon, nvidia-smi, .env, approved model, and manifest
R03 docker compose --env-file .env config --format json; validate selected ASR backend settings
R04 python -m app.asr_artifacts verify --model-dir <model> --manifest <manifest>
R05 docker compose --env-file .env build qwen-asr-api
R06 docker run qwen-asr-api:latest pinned runtime contract and Silero checksum checks
R07 nvidia-smi -L; docker run --rm --gpus all qwen-asr-api:latest nvidia-smi -L
R08 docker compose run --rm qwen-asr-api selected real ASR and VAD warmup
EOF
}

print_live_plan() {
  print_release_plan
  cat <<'EOF'
L01 API_KEY=<runtime-env> BASE_URL=<live> scripts/smoke_asr.sh
L02 stream_asr_client.py <zh-audio> --language zh --chunk-ms 200 --realtime --verify-protocol
L02 stream_asr_client.py <zh-audio> --language zh --chunk-ms 500 --realtime --verify-protocol
L02 stream_asr_client.py <ja-audio> --language ja --chunk-ms 200 --realtime --verify-protocol
L02 stream_asr_client.py <ja-audio> --language ja --chunk-ms 500 --realtime --verify-protocol
L03 concurrent zh+ja 200 ms strict streams
L04 enforce overhead plus zero-failure GPU coverage (4 samples across 0.75s) and GPU memory limit
EOF
}

print_deployed_live_plan() {
  cat <<'EOF'
D01 validate protected release receipt against clean SHA, current config/model/manifest, release evidence, and the one running image ID
L01 API_KEY=<runtime-env> BASE_URL=<live> scripts/smoke_asr.sh
L02 run the strict zh/ja 200/500 ms speech matrix against the deployed image
L03 run concurrent zh+ja 200 ms strict streams against the deployed image
L04 enforce overhead plus zero-failure GPU coverage and the GPU memory limit
No image build, Compose run, warmup container, or service recreation occurs.
EOF
}

print_plan() {
  echo "DRY RUN: no prerequisites checked and no commands executed."
  case "${MODE}" in
    commit) print_commit_plan ;;
    release) print_release_plan ;;
    live) print_live_plan ;;
    deployed-live) print_deployed_live_plan ;;
  esac
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    --list-gates)
      LIST_GATES=1
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${MODE}" ]]; then
        echo "Only one mode may be selected." >&2
        exit 2
      fi
      MODE="$1"
      ;;
  esac
  shift
done

if [[ -n "${MODE}" && "${MODE}" != "commit" && "${MODE}" != "release" && "${MODE}" != "live" && "${MODE}" != "deployed-live" ]]; then
  echo "Unknown mode: ${MODE}" >&2
  usage >&2
  exit 2
fi
if ((LIST_GATES)); then
  list_gates
  exit 0
fi
if [[ -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi
if ((DRY_RUN)); then
  print_plan
  exit 0
fi

cleanup() {
  local attempt
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    if [[ -n "${GPU_MONITOR_STATE}" ]]; then
      touch "${GPU_MONITOR_STATE}/stop" 2>/dev/null || true
    fi
    for attempt in {1..60}; do
      if ! kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.05
    done
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TEMP_DIR}" ]]; then
    rm -rf -- "${TEMP_DIR}"
  fi
}
trap cleanup EXIT HUP INT TERM

ensure_temp_dir() {
  if [[ -n "${TEMP_DIR}" ]]; then
    return
  fi
  umask 077
  TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/verify-asr-release.XXXXXX")"
}

select_python() {
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
}

add_missing() {
  MISSING+=("$1")
}

collect_commit_prerequisites() {
  local python_sentinel=""
  local pytest_sentinel=""
  if [[ -n "${ASR_VERIFY_PYTHON:-}" ]]; then
    add_missing "ASR_VERIFY_PYTHON is unsupported; use the repository environment"
  fi
  command -v git >/dev/null 2>&1 || add_missing "git"
  command -v bash >/dev/null 2>&1 || add_missing "bash"
  select_python
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || add_missing "Python (${PYTHON_BIN})"
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    python_sentinel="$("${PYTHON_BIN}" -c 'print("ASR_VERIFY_PYTHON_OK")' 2>/dev/null)" || true
    if [[ "${python_sentinel}" != "ASR_VERIFY_PYTHON_OK" ]]; then
      add_missing "functional Python interpreter at ${PYTHON_BIN}"
    fi
    pytest_sentinel="$("${PYTHON_BIN}" -c 'import pytest; print("ASR_VERIFY_PYTEST_OK")' 2>/dev/null)" || true
    if [[ "${pytest_sentinel}" != "ASR_VERIFY_PYTEST_OK" ]]; then
      add_missing "functional pytest for ${PYTHON_BIN}"
    fi
  fi
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || add_missing "Git worktree"
}

collect_release_prerequisites() {
  local env_file="${ASR_RELEASE_ENV_FILE:-${ROOT_DIR}/.env}"
  local model_dir="${ASR_RELEASE_MODEL_DIR:-}"
  local manifest="${ASR_RELEASE_MANIFEST:-}"
  local expected_env="${ROOT_DIR}/.env"
  local model_root="${ROOT_DIR}/models"
  local model_real
  local manifest_real
  local root_real

  if [[ -z "${model_dir}" ]]; then
    add_missing "ASR_RELEASE_MODEL_DIR"
  elif [[ ! -d "${model_dir}" || -L "${model_dir}" ]]; then
    add_missing "regular model directory at ASR_RELEASE_MODEL_DIR"
  fi
  if [[ -z "${manifest}" ]]; then
    add_missing "ASR_RELEASE_MANIFEST"
  elif [[ ! -f "${manifest}" || -L "${manifest}" ]]; then
    add_missing "regular manifest file at ASR_RELEASE_MANIFEST"
  fi
  if [[ -d "${model_dir}" && ! -L "${model_dir}" && -f "${manifest}" && ! -L "${manifest}" ]]; then
    if [[ ! -d "${model_root}" ]]; then
      add_missing "repository models directory containing release assets"
    else
      model_real="$(cd "${model_dir}" && pwd)"
      manifest_real="$(cd "$(dirname "${manifest}")" && pwd)/$(basename "${manifest}")"
      root_real="$(cd "${model_root}" && pwd)"
      if [[ "${model_real}" != "${root_real}/"* || "${manifest_real}" != "${root_real}/"* ]]; then
        add_missing "release model and manifest below ${model_root}"
      fi
    fi
  fi
  if [[ ! -f "${env_file}" || -L "${env_file}" ]]; then
    add_missing "regular release environment file at ASR_RELEASE_ENV_FILE"
  elif [[ "$(cd "$(dirname "${env_file}")" && pwd)/$(basename "${env_file}")" != "${expected_env}" ]]; then
    add_missing "ASR_RELEASE_ENV_FILE must be ${expected_env} for the Compose env_file mapping"
  fi
  [[ -f "Dockerfile.asr" && -f "docker-compose.yml" ]] || add_missing "ASR Docker and Compose definitions"
  command -v docker >/dev/null 2>&1 || add_missing "docker"
  if command -v docker >/dev/null 2>&1; then
    docker compose version >/dev/null 2>&1 || add_missing "Docker Compose plugin"
    docker info >/dev/null 2>&1 || add_missing "running Docker daemon"
  fi
  command -v nvidia-smi >/dev/null 2>&1 || add_missing "nvidia-smi"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L >/dev/null 2>&1 || add_missing "accessible NVIDIA GPU"
  fi
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all 2>/dev/null)" ]]; then
    add_missing "clean Git checkout"
  fi
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "$1" != "0" && "$1" != "0.0" ]]
}

collect_live_prerequisites() {
  local base_url="${ASR_LIVE_BASE_URL:-}"
  local ws_url="${ASR_LIVE_WS_URL:-}"
  local zh_audio="${ASR_LIVE_ZH_AUDIO:-}"
  local ja_audio="${ASR_LIVE_JA_AUDIO:-}"

  [[ "${base_url}" =~ ^https?:// ]] || add_missing "ASR_LIVE_BASE_URL using http or https"
  [[ "${ws_url}" =~ ^wss?:// ]] || add_missing "ASR_LIVE_WS_URL using ws or wss"
  [[ -n "${ASR_LIVE_API_KEY:-}" ]] || add_missing "ASR_LIVE_API_KEY"
  if [[ -z "${zh_audio}" || ! -f "${zh_audio}" || -L "${zh_audio}" || ! -s "${zh_audio}" ]]; then
    add_missing "regular nonempty Chinese speech file at ASR_LIVE_ZH_AUDIO"
  fi
  if [[ -z "${ja_audio}" || ! -f "${ja_audio}" || -L "${ja_audio}" || ! -s "${ja_audio}" ]]; then
    add_missing "regular nonempty Japanese speech file at ASR_LIVE_JA_AUDIO"
  fi
  command -v curl >/dev/null 2>&1 || add_missing "curl"
  command -v ffmpeg >/dev/null 2>&1 || add_missing "ffmpeg"
  command -v ffprobe >/dev/null 2>&1 || add_missing "ffprobe"
  if command -v ffprobe >/dev/null 2>&1; then
    if [[ -n "${zh_audio}" && -f "${zh_audio}" && ! -L "${zh_audio}" && -s "${zh_audio}" ]]; then
      is_positive_number "$(audio_duration "${zh_audio}" 2>/dev/null || true)" \
        || add_missing "decodable positive-duration Chinese speech audio"
    fi
    if [[ -n "${ja_audio}" && -f "${ja_audio}" && ! -L "${ja_audio}" && -s "${ja_audio}" ]]; then
      is_positive_number "$(audio_duration "${ja_audio}" 2>/dev/null || true)" \
        || add_missing "decodable positive-duration Japanese speech audio"
    fi
  fi
  if ! is_positive_number "${ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS:-}"; then
    add_missing "positive ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS"
  fi
  if [[ ! "${ASR_LIVE_MAX_GPU_MEMORY_MIB:-}" =~ ^[1-9][0-9]*$ ]]; then
    add_missing "positive integer ASR_LIVE_MAX_GPU_MEMORY_MIB"
  fi
  if [[ ! "${ASR_LIVE_GPU_INDEX:-0}" =~ ^[0-9]+$ ]]; then
    add_missing "nonnegative integer ASR_LIVE_GPU_INDEX"
  fi
}

collect_deployed_live_prerequisites() {
  local receipt="${ASR_DEPLOYED_RELEASE_RECEIPT:-}"
  if [[ -z "${receipt}" || ! -f "${receipt}" || -L "${receipt}" ]]; then
    add_missing "regular protected receipt at ASR_DEPLOYED_RELEASE_RECEIPT"
  fi
  if [[ ! -f "${ROOT_DIR}/scripts/asr_deploy_receipt.py" || -L "${ROOT_DIR}/scripts/asr_deploy_receipt.py" ]]; then
    add_missing "ASR deployment receipt validator"
  fi
}

preflight() {
  local requested_mode="$1"
  MISSING=()
  collect_commit_prerequisites
  if [[ "${requested_mode}" == "release" || "${requested_mode}" == "live" || "${requested_mode}" == "deployed-live" ]]; then
    collect_release_prerequisites
  fi
  if [[ "${requested_mode}" == "live" || "${requested_mode}" == "deployed-live" ]]; then
    collect_live_prerequisites
  fi
  if [[ "${requested_mode}" == "deployed-live" ]]; then
    collect_deployed_live_prerequisites
  fi
  if ((${#MISSING[@]})); then
    echo "Missing ${requested_mode} prerequisites:" >&2
    printf '  - %s\n' "${MISSING[@]}" >&2
    return 1
  fi
}

gate() {
  echo "==> [$1] $2"
}

scan_secrets() {
  local pattern
  local status
  local matches="${TEMP_DIR}/secret-paths"
  pattern='AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|sk-(proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----'
  : >"${matches}"
  status=0
  git grep -I -l -E "${pattern}" -- . >>"${matches}" || status=$?
  if ((status > 1)); then
    echo "Secret scan failed while reading the tracked worktree." >&2
    return 1
  fi
  status=0
  git grep --cached -I -l -E "${pattern}" -- . >>"${matches}" || status=$?
  if ((status > 1)); then
    echo "Secret scan failed while reading the index." >&2
    return 1
  fi
  sort -u -o "${matches}" "${matches}"
  if [[ -s "${matches}" ]]; then
    echo "High-confidence credential material found in:" >&2
    sed 's/^/  - /' "${matches}" >&2
    return 1
  fi
}

scan_forbidden_artifacts() {
  local path
  local lower
  local size
  local limit="${ASR_VERIFY_MAX_STAGED_BYTES:-1048576}"
  local forbidden="${TEMP_DIR}/forbidden-paths"
  local binary="${TEMP_DIR}/binary-delta"
  local large="${TEMP_DIR}/large-delta"
  [[ "${limit}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ASR_VERIFY_MAX_STAGED_BYTES must be a positive integer." >&2
    return 1
  }
  : >"${forbidden}"
  while IFS= read -r -d '' path; do
    lower="${path,,}"
    case "/${lower}" in
      */superpowers/*|*/docs/superpowers/*|*/__pycache__/*|*/.pytest_cache/*|*/htmlcov/*|*/.env|*.pyc|*.pyo|*.html|*.htm|*.onnx|*.safetensors|*.pt|*.pth|*.ckpt|*.gguf|*.bin|*.wav|*.wave|*.mp3|*.flac|*.ogg|*.m4a|*.aac)
        printf '%s\n' "${path}" >>"${forbidden}"
        ;;
    esac
  done < <(git ls-files -z)
  if [[ -s "${forbidden}" ]]; then
    echo "Forbidden tracked artifact paths found:" >&2
    sed 's/^/  - /' "${forbidden}" >&2
    return 1
  fi

  git diff --cached --numstat --diff-filter=ACMR | awk '$1 == "-" || $2 == "-"' >"${binary}"
  if [[ -s "${binary}" ]]; then
    echo "Binary staged deltas are not allowed:" >&2
    sed 's/^/  - /' "${binary}" >&2
    return 1
  fi

  : >"${large}"
  while IFS= read -r -d '' path; do
    size="$(git cat-file -s ":${path}")"
    if ((size > limit)); then
      printf '%s (%s bytes)\n' "${path}" "${size}" >>"${large}"
    fi
  done < <(git diff --cached --name-only -z --diff-filter=ACMR)
  if [[ -s "${large}" ]]; then
    echo "Large staged deltas exceed ${limit} bytes:" >&2
    sed 's/^/  - /' "${large}" >&2
    return 1
  fi
}

run_commit_gates() {
  local script_path
  local before_status
  local after_status
  local mock_api_key
  preflight commit
  ensure_temp_dir
  before_status="${TEMP_DIR}/git-status.before"
  after_status="${TEMP_DIR}/git-status.after"
  git status --porcelain=v1 --untracked-files=all >"${before_status}"

  gate C04 "Git whitespace checks"
  git diff --check
  git diff --cached --check

  gate C05 "High-confidence secret scan"
  scan_secrets

  gate C06 "Forbidden path and staged artifact checks"
  scan_forbidden_artifacts

  gate C03 "Tracked shell syntax"
  while IFS= read -r -d '' script_path; do
    bash -n "${script_path}"
  done < <(git ls-files -z '*.sh')

  gate C02 "Python compileall redirected under temporary storage"
  PYTHONPYCACHEPREFIX="${TEMP_DIR}/pycache" \
    "${PYTHON_BIN}" -m compileall -q app tests scripts

  gate C01 "Full explicit-mock pytest suite"
  mock_api_key="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_hex(32))')"
  PYTHONDONTWRITEBYTECODE=1 \
    MODEL_BACKEND=mock \
    ASR_BACKEND=mock \
    ASR_STREAM_MODE=chunked \
    ASR_REQUIRE_MODEL_MANIFEST=false \
    ASR_MODEL_MANIFEST_PATH= \
    ASR_VLLM_GPU_MEMORY_UTILIZATION=0.8 \
    ASR_VLLM_MAX_MODEL_LEN=65536 \
    TTS_BACKEND=mock \
    API_KEY="${mock_api_key}" \
    "${PYTHON_BIN}" -m pytest tests -q -p no:cacheprovider

  git status --porcelain=v1 --untracked-files=all >"${after_status}"
  if ! cmp -s "${before_status}" "${after_status}"; then
    echo "Commit verification changed repository status." >&2
    diff -u "${before_status}" "${after_status}" >&2 || true
    return 1
  fi
}

validate_compose_config() {
  local rendered="${TEMP_DIR}/compose-config.json"
  local model_dir="${ASR_RELEASE_MODEL_DIR}"
  local manifest="${ASR_RELEASE_MANIFEST}"
  local model_root="${ROOT_DIR}/models"
  local model_real
  local manifest_real
  local root_real

  docker compose --env-file "${ASR_RELEASE_ENV_FILE:-${ROOT_DIR}/.env}" \
    config --format json >"${rendered}"
  model_real="$(cd "${model_dir}" && pwd)"
  manifest_real="$(cd "$(dirname "${manifest}")" && pwd)/$(basename "${manifest}")"
  root_real="$(cd "${model_root}" && pwd)"
  PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" - "${rendered}" "${root_real}" "${model_real}" "${manifest_real}" <<'PY'
import json
import sys
from pathlib import Path

from app.config import PRODUCTION_API_KEY_MIN_LENGTH, PRODUCTION_API_KEY_PLACEHOLDERS

config_path, model_root_raw, model_raw, manifest_raw = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
try:
    service = config["services"]["qwen-asr-api"]
except (KeyError, TypeError) as exc:
    raise SystemExit("Compose config has no qwen-asr-api service") from exc
environment = service.get("environment") or {}
if isinstance(environment, list):
    environment = dict(item.split("=", 1) for item in environment if "=" in item)
backend = str(environment.get("ASR_BACKEND", "")).lower()
stream_contracts = {
    "qwen_vllm": "stateful",
    "faster_whisper": "rolling",
}
if backend not in stream_contracts:
    raise SystemExit(f"Compose ASR_BACKEND is unsupported for release: {backend or '<missing>'}")
required = {
    "ASR_STREAM_MODE": stream_contracts[backend],
    "ASR_REQUIRE_MODEL_MANIFEST": "true",
    "ASR_EAGER_LOAD": "true",
    "ASR_FILE_TRANSCRIBE_ENABLED": "false",
}
if backend == "faster_whisper":
    required.update({
        "ASR_FASTER_WHISPER_COMPUTE_TYPE": "float16",
        "ASR_FASTER_WHISPER_BATCH_SIZE": "4",
        "ASR_FASTER_WHISPER_PARTIAL_BEAM_SIZE": "1",
        "ASR_FASTER_WHISPER_FINAL_BEAM_SIZE": "5",
        "ASR_FASTER_WHISPER_TASK": "transcribe",
    })
for name, expected in required.items():
    actual = str(environment.get(name, "")).lower()
    if actual != expected:
        raise SystemExit(f"Compose {name} must be {expected}, got {actual or '<missing>'}")
api_key = str(environment.get("API_KEY", ""))
if (
    len(api_key.strip()) < PRODUCTION_API_KEY_MIN_LENGTH
    or api_key.strip().lower() in PRODUCTION_API_KEY_PLACEHOLDERS
):
    raise SystemExit("Compose API_KEY must be a non-placeholder production secret")
command = [str(item) for item in service.get("command") or []]
if "--workers" not in command or command[command.index("--workers") + 1] != "1":
    raise SystemExit("qwen-asr-api must run exactly one Uvicorn worker")
volumes = service.get("volumes") or []
if not any(
    (isinstance(item, dict) and item.get("target") == "/models" and item.get("read_only"))
    or (isinstance(item, str) and ":/models:ro" in item)
    for item in volumes
):
    raise SystemExit("qwen-asr-api must mount /models read-only")
model_root = Path(model_root_raw)
model = Path(model_raw)
manifest = Path(manifest_raw)
try:
    model_relative = model.relative_to(model_root)
    manifest_relative = manifest.relative_to(model_root)
except ValueError as exc:
    raise SystemExit("Release model and manifest must be below the repository models directory") from exc
expected_model = "/models/" + model_relative.as_posix()
expected_manifest = "/models/" + manifest_relative.as_posix()
if environment.get("ASR_MODEL_ID") != expected_model:
    raise SystemExit("Compose ASR_MODEL_ID does not match ASR_RELEASE_MODEL_DIR")
if environment.get("ASR_MODEL_MANIFEST_PATH") != expected_manifest:
    raise SystemExit("Compose ASR_MODEL_MANIFEST_PATH does not match ASR_RELEASE_MANIFEST")
PY
}

validate_deployed_release_receipt() {
  local container_ids
  local container_count
  local container_id
  local running_image_id
  local candidate_sha
  local env_file="${ASR_RELEASE_ENV_FILE:-${ROOT_DIR}/.env}"

  container_ids="$(docker compose --env-file "${env_file}" ps -q qwen-asr-api)"
  container_count="$(printf '%s\n' "${container_ids}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  if [[ "${container_count}" != "1" ]]; then
    echo "Deployed-live requires exactly one running qwen-asr-api container; found ${container_count}." >&2
    return 1
  fi
  container_id="${container_ids}"
  running_image_id="$(docker inspect --format '{{.Image}}' "${container_id}")"
  candidate_sha="$(git rev-parse HEAD)"

  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_deploy_receipt.py \
    validate-receipt \
    --kind release \
    --receipt "${ASR_DEPLOYED_RELEASE_RECEIPT}" \
    --repository "${ROOT_DIR}" \
    --candidate-sha "${candidate_sha}" \
    --image-id "${running_image_id}" \
    --env-file "${env_file}" \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"
  validate_compose_config
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m app.asr_artifacts verify \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"
}

run_release_gates() {
  local env_file="${ASR_RELEASE_ENV_FILE:-${ROOT_DIR}/.env}"
  preflight release
  run_commit_gates
  ensure_temp_dir

  gate R01 "Clean release checkout"
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]

  gate R03 "Rendered Docker Compose production config"
  validate_compose_config

  gate R04 "Operator-approved model manifest"
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m app.asr_artifacts verify \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"

  gate R05 "Build qwen-asr-api image"
  docker compose --env-file "${env_file}" build qwen-asr-api
  docker image inspect qwen-asr-api:latest >/dev/null

  gate R06 "Pinned ASR runtime contracts inside image"
  docker run --rm \
    --entrypoint python \
    --volume "${ROOT_DIR}/scripts/check_qwen_streaming_contract.py:/tmp/check_qwen_streaming_contract.py:ro" \
    qwen-asr-api:latest \
    /tmp/check_qwen_streaming_contract.py
  docker run --rm --entrypoint python qwen-asr-api:latest -c \
    'import importlib.metadata as m; assert m.version("faster-whisper") == "1.2.1"; assert m.version("ctranslate2") == "4.8.1"'
  docker run --rm --entrypoint python qwen-asr-api:latest -c \
    'import hashlib, pathlib; p=pathlib.Path("/opt/asr-assets/silero_vad.onnx"); expected="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"; actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual == expected, actual'

  gate R07 "Host and container NVIDIA GPU access"
  nvidia-smi -L
  docker run --rm --gpus all --entrypoint nvidia-smi qwen-asr-api:latest -L

  gate R08 "One-shot selected ASR, manifest, VAD, and streaming warmup"
  docker compose --env-file "${env_file}" run --rm --no-deps \
    --entrypoint python qwen-asr-api -c \
    'import asyncio
from app.asr_gateway import _default_runtime
from app.asr_vad import create_vad_endpoint_detector
from app.config import Settings
async def verify():
    settings = Settings()
    create_vad_endpoint_detector(settings)
    runtime = _default_runtime()
    await runtime.start()
    try:
        snapshot = await runtime.adapters["local"].snapshot()
        assert snapshot["ready"] and snapshot["accepting"]
    finally:
        await runtime.close()
asyncio.run(verify())'
}

audio_duration() {
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1"
}

check_stream_overhead() {
  local audio_file="$1"
  local started="$2"
  local label="$3"
  local duration
  duration="$(audio_duration "${audio_file}")"
  "${PYTHON_BIN}" - "${started}" "${duration}" "${ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS}" "${label}" <<'PY'
import sys
import time

started, duration, maximum = map(float, sys.argv[1:4])
label = sys.argv[4]
elapsed = time.monotonic() - started
overhead = max(0.0, elapsed - duration)
print(f"{label}: elapsed={elapsed:.2f}s audio={duration:.2f}s overhead={overhead:.2f}s")
if overhead > maximum:
    raise SystemExit(
        f"{label} exceeded stream overhead limit: {overhead:.2f}s > {maximum:.2f}s"
    )
PY
}

run_live_client() {
  local audio_file="$1"
  local language="$2"
  local chunk_ms="$3"
  local label="${language}-${chunk_ms}ms"
  local started
  started="$("${PYTHON_BIN}" -c 'import time; print(time.monotonic())')"
  API_KEY="${ASR_LIVE_API_KEY}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${PYTHON_BIN}" scripts/stream_asr_client.py "${audio_file}" \
      --url "${ASR_LIVE_WS_URL}" \
      --language "${language}" \
      --chunk-ms "${chunk_ms}" \
      --realtime \
      --verify-protocol
  check_stream_overhead "${audio_file}" "${started}" "${label}"
}

start_gpu_monitor() {
  GPU_MONITOR_STATE="${TEMP_DIR}/gpu-monitor"
  mkdir -p "${GPU_MONITOR_STATE}"
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_gpu_monitor.py record \
    --state-dir "${GPU_MONITOR_STATE}" \
    --gpu-index "${ASR_LIVE_GPU_INDEX:-0}" \
    --interval-seconds 0.25 &
  GPU_MONITOR_PID=$!
}

stop_and_validate_gpu_monitor() {
  local monitor_status=0
  touch "${GPU_MONITOR_STATE}/stop"
  wait "${GPU_MONITOR_PID}" || monitor_status=$?
  GPU_MONITOR_PID=""
  if ((monitor_status != 0)); then
    echo "GPU monitor process failed with status ${monitor_status}." >&2
    return 1
  fi
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_gpu_monitor.py validate \
    --state-dir "${GPU_MONITOR_STATE}" \
    --maximum-memory-mib "${ASR_LIVE_MAX_GPU_MEMORY_MIB}" \
    --minimum-samples 4 \
    --minimum-span-seconds 0.75
}

run_concurrent_live_gate() {
  local zh_log="${TEMP_DIR}/concurrent-zh.log"
  local ja_log="${TEMP_DIR}/concurrent-ja.log"
  local zh_pid
  local ja_pid
  local zh_status=0
  local ja_status=0

  (run_live_client "${ASR_LIVE_ZH_AUDIO}" zh 200) >"${zh_log}" 2>&1 &
  zh_pid=$!
  (run_live_client "${ASR_LIVE_JA_AUDIO}" ja 200) >"${ja_log}" 2>&1 &
  ja_pid=$!
  wait "${zh_pid}" || zh_status=$?
  wait "${ja_pid}" || ja_status=$?
  sed 's/^/[concurrent-zh] /' "${zh_log}"
  sed 's/^/[concurrent-ja] /' "${ja_log}"
  if ((zh_status != 0 || ja_status != 0)); then
    echo "Concurrent live gate failed: zh=${zh_status}, ja=${ja_status}" >&2
    return 1
  fi
}

run_live_acceptance_gates() {
  ensure_temp_dir
  start_gpu_monitor

  gate L01 "Deployed readiness and WebSocket lifecycle smoke"
  API_KEY="${ASR_LIVE_API_KEY}" \
    BASE_URL="${ASR_LIVE_BASE_URL}" \
    AUDIO_FILE="" \
    LANGUAGE="" \
    EXPECT_ASR_STREAM_MODE="${ASR_LIVE_EXPECT_STREAM_MODE:-stateful}" \
    EXPECT_ASR_BACKEND="${ASR_LIVE_EXPECT_BACKEND:-qwen_vllm}" \
    EXPECT_ASR_STABLE_COMMIT_ENABLED="${ASR_LIVE_EXPECT_STABLE_COMMIT_ENABLED:-false}" \
    scripts/smoke_asr.sh

  gate L02 "Strict zh and ja 200/500 ms speech matrix"
  run_live_client "${ASR_LIVE_ZH_AUDIO}" zh 200
  run_live_client "${ASR_LIVE_ZH_AUDIO}" zh 500
  run_live_client "${ASR_LIVE_JA_AUDIO}" ja 200
  run_live_client "${ASR_LIVE_JA_AUDIO}" ja 500

  gate L03 "Concurrent zh and ja strict streams"
  run_concurrent_live_gate

  gate L04 "Configured live GPU memory limit"
  stop_and_validate_gpu_monitor
}

run_live_gates() {
  preflight live
  run_release_gates
  run_live_acceptance_gates
}

run_deployed_live_gates() {
  preflight deployed-live
  gate D01 "Release receipt bound to the one deployed image and artifact set"
  validate_deployed_release_receipt
  run_live_acceptance_gates
}

case "${MODE}" in
  commit) run_commit_gates ;;
  release) run_release_gates ;;
  live) run_live_gates ;;
  deployed-live) run_deployed_live_gates ;;
esac

echo "ASR ${MODE} verification passed."
