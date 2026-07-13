#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
RUN_LIVE=1
PYTHON_BIN=""
CANDIDATE_SHA=""
RELEASE_ID=""
RELEASE_IMAGE_ID=""
LIVE_VERIFIED_IMAGE_ID=""
RELEASE_IMAGE_REF=""
ROLLBACK_IMAGE_ID=""
ROLLBACK_IMAGE_REF=""
BACKUP_RUN_DIR=""
ENV_BACKUP=""
MANIFEST_BACKUP=""
ENV_SHA256=""
MANIFEST_SHA256=""
CUTOVER_STARTED=0
DEPLOYMENT_COMPLETE=0
SNAPSHOT_READY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_asr_cloud.sh [--dry-run] [--skip-live]

Build, verify, deploy, and live-verify one production ASR image. Full
release/deploy/live verification is the fail-closed default.

Options:
  --dry-run    Print the ordered workflow without checking prerequisites or changing anything.
  --skip-live  Stop after release verification, exact-image cutover, readiness, and smoke.
  -h, --help   Show this help.

Release environment (defaults shown):
  ASR_RELEASE_ENV_FILE       <repository>/.env
  ASR_RELEASE_MODEL_DIR      <repository>/models/Qwen3-ASR-1.7B-hf
  ASR_RELEASE_MANIFEST       <repository>/models/Qwen3-ASR-1.7B-hf.manifest.json

Live environment:
  ASR_LIVE_BASE_URL                    Default: http://127.0.0.1:8002
  ASR_LIVE_WS_URL                      Default: ws://127.0.0.1:8002/v1/transcribe/stream
  ASR_LIVE_ZH_AUDIO                    External Chinese speech audio
  ASR_LIVE_JA_AUDIO                    External Japanese speech audio
  ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS Approved positive SLO threshold
  ASR_LIVE_MAX_GPU_MEMORY_MIB          Approved positive VRAM threshold
  ASR_LIVE_GPU_INDEX                   Default: 0
  ASR_LIVE_API_KEY                     Environment only; otherwise requested with hidden input

Deployment environment:
  ASR_DEPLOY_EVIDENCE_DIR              Default: /secure/asr-release-evidence
  ASR_DEPLOY_BACKUP_DIR                Default: /secure/asr-release-backup
  ASR_DEPLOY_LOCAL_BASE_URL            Default: http://127.0.0.1:8002
  ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS     Default: 600
  ASR_DEPLOY_HEALTH_INTERVAL_SECONDS    Default: 5

The wrapper never creates a model manifest and never cleans, resets, stashes,
checks out, or pulls Git state. External release assets must already exist.
EOF
}

print_plan() {
  cat <<'EOF'
DRY RUN: no prerequisites checked and no commands executed.
01 validate clean committed checkout and external release/live inputs
02 prepare repository .venv with pinned requirements-dev.txt when needed
03 snapshot current deployment image and back up .env plus approved manifest
04 run scripts/verify_asr_release.sh release with protected evidence
05 capture and tag the exact release-verified image ID
06 deploy the exact release-verified image without rebuilding
07 verify local readiness and WebSocket smoke
08 run scripts/verify_asr_release.sh live with protected evidence
09 reassert the accepted image ID and unchanged config/model/manifest
On any failure or interruption after step 06: atomically restore the previous
image, .env, and approved manifest, verify the unchanged model, then rerun local
readiness and WebSocket smoke while preserving the original exit status.
EOF
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
    --skip-live)
      RUN_LIVE=0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ((DRY_RUN)); then
  print_plan
  if ((!RUN_LIVE)); then
    echo "--skip-live selected: step 08 will be skipped."
  fi
  exit 0
fi

cd "${ROOT_DIR}"
umask 077

export ASR_RELEASE_ENV_FILE="${ASR_RELEASE_ENV_FILE:-${ROOT_DIR}/.env}"
export ASR_RELEASE_MODEL_DIR="${ASR_RELEASE_MODEL_DIR:-${ROOT_DIR}/models/Qwen3-ASR-1.7B-hf}"
export ASR_RELEASE_MANIFEST="${ASR_RELEASE_MANIFEST:-${ROOT_DIR}/models/Qwen3-ASR-1.7B-hf.manifest.json}"
export ASR_LIVE_BASE_URL="${ASR_LIVE_BASE_URL:-http://127.0.0.1:8002}"
export ASR_LIVE_WS_URL="${ASR_LIVE_WS_URL:-ws://127.0.0.1:8002/v1/transcribe/stream}"
export ASR_LIVE_GPU_INDEX="${ASR_LIVE_GPU_INDEX:-0}"
ASR_DEPLOY_EVIDENCE_DIR="${ASR_DEPLOY_EVIDENCE_DIR:-/secure/asr-release-evidence}"
ASR_DEPLOY_BACKUP_DIR="${ASR_DEPLOY_BACKUP_DIR:-/secure/asr-release-backup}"
ASR_DEPLOY_LOCAL_BASE_URL="${ASR_DEPLOY_LOCAL_BASE_URL:-http://127.0.0.1:8002}"
ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS="${ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS:-600}"
ASR_DEPLOY_HEALTH_INTERVAL_SECONDS="${ASR_DEPLOY_HEALTH_INTERVAL_SECONDS:-5}"

fail() {
  echo "ASR cloud deployment failed: $*" >&2
  return 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "$1" != "0" && "$1" != "0.0" ]]
}

require_regular_file() {
  [[ -f "$1" && ! -L "$1" ]] || fail "required regular file is missing or a symlink: $1"
}

require_external_directory() {
  local requested="$1"
  local resolved
  if [[ -L "${requested}" ]]; then
    fail "protected directory must not be a symlink: ${requested}"
    return
  fi
  resolved="$(realpath -m -- "${requested}")"
  case "${resolved}" in
    "${ROOT_DIR}"|"${ROOT_DIR}"/*)
      fail "protected directory must be outside the repository: ${requested}"
      ;;
  esac
}

validate_checkout_and_assets() {
  local git_root
  require_command git
  git_root="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || fail "run from a committed Git worktree"
  [[ "$(realpath -m -- "${git_root}")" == "${ROOT_DIR}" ]] \
    || fail "script and Git worktree roots do not match"
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    fail "release requires a clean Git checkout; resolve changes manually (no Git state was modified)"
  fi
  CANDIDATE_SHA="$(git rev-parse HEAD)"
  [[ "${CANDIDATE_SHA}" =~ ^[0-9a-fA-F]{40}$ ]] || fail "could not resolve a full candidate commit SHA"

  require_regular_file "${ROOT_DIR}/requirements-dev.txt"
  require_regular_file "${ROOT_DIR}/scripts/verify_asr_release.sh"
  require_regular_file "${ROOT_DIR}/scripts/smoke_asr.sh"
  [[ -x "${ROOT_DIR}/scripts/verify_asr_release.sh" ]] || fail "release verifier is not executable"
  [[ -x "${ROOT_DIR}/scripts/smoke_asr.sh" ]] || fail "ASR smoke script is not executable"
  require_regular_file "${ASR_RELEASE_ENV_FILE}"
  [[ "$(realpath -m -- "${ASR_RELEASE_ENV_FILE}")" == "${ROOT_DIR}/.env" ]] \
    || fail "ASR_RELEASE_ENV_FILE must be ${ROOT_DIR}/.env"
  [[ -d "${ASR_RELEASE_MODEL_DIR}" && ! -L "${ASR_RELEASE_MODEL_DIR}" ]] \
    || fail "ASR_RELEASE_MODEL_DIR must be an existing non-symlink directory"
  require_regular_file "${ASR_RELEASE_MANIFEST}"
  case "$(realpath -m -- "${ASR_RELEASE_MODEL_DIR}")" in
    "${ROOT_DIR}/models"/*) ;;
    *) fail "ASR_RELEASE_MODEL_DIR must be below ${ROOT_DIR}/models" ;;
  esac
  case "$(realpath -m -- "${ASR_RELEASE_MANIFEST}")" in
    "${ROOT_DIR}/models"/*) ;;
    *) fail "ASR_RELEASE_MANIFEST must be below ${ROOT_DIR}/models" ;;
  esac

  require_external_directory "${ASR_DEPLOY_EVIDENCE_DIR}"
  require_external_directory "${ASR_DEPLOY_BACKUP_DIR}"
}

validate_live_inputs() {
  [[ "${ASR_LIVE_BASE_URL}" =~ ^https?:// ]] \
    || fail "ASR_LIVE_BASE_URL must use http or https"
  [[ "${ASR_LIVE_WS_URL}" =~ ^wss?:// ]] \
    || fail "ASR_LIVE_WS_URL must use ws or wss"
  require_regular_file "${ASR_LIVE_ZH_AUDIO:-}"
  [[ -s "${ASR_LIVE_ZH_AUDIO}" ]] || fail "ASR_LIVE_ZH_AUDIO must be nonempty"
  require_regular_file "${ASR_LIVE_JA_AUDIO:-}"
  [[ -s "${ASR_LIVE_JA_AUDIO}" ]] || fail "ASR_LIVE_JA_AUDIO must be nonempty"
  is_positive_number "${ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS:-}" \
    || fail "ASR_LIVE_MAX_STREAM_OVERHEAD_SECONDS must be a positive approved threshold"
  [[ "${ASR_LIVE_MAX_GPU_MEMORY_MIB:-}" =~ ^[1-9][0-9]*$ ]] \
    || fail "ASR_LIVE_MAX_GPU_MEMORY_MIB must be a positive approved integer threshold"
  [[ "${ASR_LIVE_GPU_INDEX}" =~ ^[0-9]+$ ]] \
    || fail "ASR_LIVE_GPU_INDEX must be a nonnegative integer"
  require_command ffmpeg
  require_command ffprobe
  is_positive_number "$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${ASR_LIVE_ZH_AUDIO}" 2>/dev/null)" \
    || fail "ASR_LIVE_ZH_AUDIO must have a positive decodable duration"
  is_positive_number "$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${ASR_LIVE_JA_AUDIO}" 2>/dev/null)" \
    || fail "ASR_LIVE_JA_AUDIO must have a positive decodable duration"
}

select_or_prepare_python() {
  local pytest_pin
  local installed=""
  pytest_pin="$(sed -nE 's/^[[:space:]]*pytest==([^[:space:]#]+)[[:space:]]*$/\1/p' requirements-dev.txt)"
  [[ -n "${pytest_pin}" && "${pytest_pin}" != *$'\n'* ]] \
    || fail "requirements-dev.txt must contain exactly one pinned pytest requirement"

  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  if [[ -x "${PYTHON_BIN}" ]]; then
    installed="$("${PYTHON_BIN}" -c 'import pytest; print(pytest.__version__)' 2>/dev/null || true)"
  fi
  if [[ "${installed}" != "${pytest_pin}" ]]; then
    require_command python3
    echo "Preparing repository .venv with pinned development requirements..."
    python3 -m venv "${ROOT_DIR}/.venv"
    "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-dev.txt"
    installed="$("${PYTHON_BIN}" -c 'import pytest; print(pytest.__version__)' 2>/dev/null || true)"
  fi
  [[ "${installed}" == "${pytest_pin}" ]] \
    || fail "repository .venv pytest version does not match requirements-dev.txt (${pytest_pin})"
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    fail "environment preparation changed tracked Git status; resolve it manually"
  fi
}

obtain_api_key() {
  if [[ -n "${ASR_LIVE_API_KEY:-}" ]]; then
    export ASR_LIVE_API_KEY
    return
  fi
  if [[ ! -t 0 ]]; then
    fail "ASR_LIVE_API_KEY is required in the environment when hidden interactive input is unavailable"
    return
  fi
  read -r -s -p "Deployed ASR API key: " ASR_LIVE_API_KEY
  echo >&2
  [[ -n "${ASR_LIVE_API_KEY}" ]] || fail "ASR_LIVE_API_KEY must not be empty"
  export ASR_LIVE_API_KEY
}

prepare_protected_directories() {
  mkdir -p -- "${ASR_DEPLOY_EVIDENCE_DIR}" "${ASR_DEPLOY_BACKUP_DIR}"
  [[ ! -L "${ASR_DEPLOY_EVIDENCE_DIR}" && ! -L "${ASR_DEPLOY_BACKUP_DIR}" ]] \
    || fail "protected evidence or backup path became a symlink"
  chmod 700 -- "${ASR_DEPLOY_EVIDENCE_DIR}" "${ASR_DEPLOY_BACKUP_DIR}"
}

compose() {
  docker compose --env-file "${ASR_RELEASE_ENV_FILE}" "$@"
}

current_container_id() {
  local ids
  local count
  ids="$(compose ps -q qwen-asr-api)"
  count="$(printf '%s\n' "${ids}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  if [[ "${count}" != 1 ]]; then
    fail "expected exactly one deployed qwen-asr-api container for rollback, found ${count}"
    return 1
  fi
  printf '%s\n' "${ids}"
}

snapshot_current_deployment() {
  local current_container
  local timestamp
  current_container="$(current_container_id)"
  ROLLBACK_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${current_container}")"
  [[ -n "${ROLLBACK_IMAGE_ID}" ]] || fail "could not inspect the current ASR image ID"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  RELEASE_ID="${CANDIDATE_SHA:0:12}-${timestamp}"
  ROLLBACK_IMAGE_REF="qwen-asr-api:rollback-${RELEASE_ID}"
  RELEASE_IMAGE_REF="qwen-asr-api:release-${RELEASE_ID}"
  BACKUP_RUN_DIR="${ASR_DEPLOY_BACKUP_DIR}/${RELEASE_ID}"
  mkdir -- "${BACKUP_RUN_DIR}"
  chmod 700 -- "${BACKUP_RUN_DIR}"
  ENV_BACKUP="${BACKUP_RUN_DIR}/release.env"
  MANIFEST_BACKUP="${BACKUP_RUN_DIR}/approved.manifest.json"
  install -m 600 -- "${ASR_RELEASE_ENV_FILE}" "${ENV_BACKUP}"
  install -m 600 -- "${ASR_RELEASE_MANIFEST}" "${MANIFEST_BACKUP}"
  ENV_SHA256="$(sha256sum -- "${ENV_BACKUP}" | awk '{print $1}')"
  MANIFEST_SHA256="$(sha256sum -- "${MANIFEST_BACKUP}" | awk '{print $1}')"
  docker tag "${ROLLBACK_IMAGE_ID}" "${ROLLBACK_IMAGE_REF}"
  {
    printf 'candidate_sha=%s\n' "${CANDIDATE_SHA}"
    printf 'rollback_image_id=%s\n' "${ROLLBACK_IMAGE_ID}"
    printf 'env_sha256=%s\n' "${ENV_SHA256}"
    printf 'manifest_sha256=%s\n' "${MANIFEST_SHA256}"
  } >"${BACKUP_RUN_DIR}/identity.txt"
  chmod 600 -- "${BACKUP_RUN_DIR}/identity.txt"
  SNAPSHOT_READY=1
}

assert_release_assets_unchanged() {
  local current_env_sha
  local current_manifest_sha
  current_env_sha="$(sha256sum -- "${ASR_RELEASE_ENV_FILE}" | awk '{print $1}')"
  current_manifest_sha="$(sha256sum -- "${ASR_RELEASE_MANIFEST}" | awk '{print $1}')"
  if [[ "${current_env_sha}" != "${ENV_SHA256}" ]]; then
    fail "release .env changed during deployment"
    return 1
  fi
  if [[ "${current_manifest_sha}" != "${MANIFEST_SHA256}" ]]; then
    fail "approved manifest changed during deployment"
    return 1
  fi
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m app.asr_artifacts verify \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"
}

run_with_evidence() {
  local evidence_file="$1"
  local command_status
  local tee_status
  local statuses
  shift
  set +e
  "$@" 2>&1 | tee "${evidence_file}"
  statuses=("${PIPESTATUS[@]}")
  command_status="${statuses[0]}"
  tee_status="${statuses[1]}"
  set -e
  chmod 600 -- "${evidence_file}" 2>/dev/null || true
  if ((command_status != 0)); then
    return "${command_status}"
  fi
  if ((tee_status != 0)); then
    echo "Could not retain protected deployment evidence: ${evidence_file}" >&2
    return 74
  fi
}

assert_running_image() {
  local expected="$1"
  local container
  local actual
  container="$(current_container_id)"
  actual="$(docker inspect --format '{{.Image}}' "${container}")"
  [[ "${actual}" == "${expected}" ]] \
    || fail "running ASR image mismatch: expected ${expected}, got ${actual:-<missing>}"
}

wait_for_readiness() {
  local deadline
  is_positive_number "${ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS}" \
    || fail "ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS must be positive"
  is_positive_number "${ASR_DEPLOY_HEALTH_INTERVAL_SECONDS}" \
    || fail "ASR_DEPLOY_HEALTH_INTERVAL_SECONDS must be positive"
  deadline=$((SECONDS + ${ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS%.*}))
  while ((SECONDS <= deadline)); do
    if curl -fsS "${ASR_DEPLOY_LOCAL_BASE_URL}/ready" >/dev/null; then
      echo "Local ASR readiness passed."
      return 0
    fi
    sleep "${ASR_DEPLOY_HEALTH_INTERVAL_SECONDS}"
  done
  fail "local readiness did not pass within ${ASR_DEPLOY_HEALTH_TIMEOUT_SECONDS}s"
}

run_local_smoke() {
  API_KEY="${ASR_LIVE_API_KEY}" \
    BASE_URL="${ASR_DEPLOY_LOCAL_BASE_URL}" \
    AUDIO_FILE="" \
    LANGUAGE="" \
    EXPECT_ASR_STREAM_MODE="stateful" \
    EXPECT_ASR_BACKEND="qwen_vllm" \
    EXPECT_ASR_STABLE_COMMIT_ENABLED="false" \
    "${ROOT_DIR}/scripts/smoke_asr.sh"
}

verify_local_deployment() {
  wait_for_readiness || return $?
  run_local_smoke
}

rollback_deployment() {
  local rollback_failed=0
  echo "Attempting atomic ASR rollback..." >&2

  if [[ -L "${ASR_RELEASE_ENV_FILE}" || -L "${ASR_RELEASE_MANIFEST}" ]]; then
    echo "Rollback refused to write through a release asset symlink." >&2
    rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    install -m 600 -- "${ENV_BACKUP}" "${ASR_RELEASE_ENV_FILE}" || rollback_failed=1
    install -m 600 -- "${MANIFEST_BACKUP}" "${ASR_RELEASE_MANIFEST}" || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    assert_release_assets_unchanged || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    docker tag "${ROLLBACK_IMAGE_ID}" qwen-asr-api:latest || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    compose up -d --force-recreate --no-deps --no-build qwen-asr-api || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    assert_running_image "${ROLLBACK_IMAGE_ID}" || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    verify_local_deployment || rollback_failed=1
  fi

  if ((rollback_failed != 0)); then
    echo "ROLLBACK FAILED: previous image/config/manifest readiness and smoke were not restored." >&2
    return 1
  fi
  echo "Rollback completed; previous image/config/manifest readiness and smoke passed." >&2
}

on_exit() {
  local original_status=$?
  local rollback_status=0
  trap - EXIT HUP INT TERM
  if ((CUTOVER_STARTED && !DEPLOYMENT_COMPLETE)); then
    if ((original_status == 0)); then
      original_status=1
    fi
    set +e
    rollback_deployment
    rollback_status=$?
    set -e
    if ((rollback_status != 0)); then
      echo "Original deployment failure status ${original_status} is preserved despite rollback failure." >&2
    fi
  elif ((SNAPSHOT_READY && !DEPLOYMENT_COMPLETE)); then
    if ((original_status == 0)); then
      original_status=1
    fi
    set +e
    docker tag "${ROLLBACK_IMAGE_ID}" qwen-asr-api:latest
    rollback_status=$?
    set -e
    if ((rollback_status != 0)); then
      echo "Could not restore qwen-asr-api:latest after the pre-cutover failure; original status ${original_status} is preserved." >&2
    fi
  fi
  exit "${original_status}"
}

handle_signal() {
  local status="$1"
  echo "Deployment interrupted; rollback will be attempted if cutover started." >&2
  exit "${status}"
}

trap on_exit EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

for command_name in realpath sed wc tr docker curl date install sha256sum awk tee chmod mkdir sleep; do
  require_command "${command_name}"
done
validate_checkout_and_assets
select_or_prepare_python
if ((RUN_LIVE)); then
  validate_live_inputs
fi
obtain_api_key
prepare_protected_directories
snapshot_current_deployment
assert_release_assets_unchanged

release_evidence="${ASR_DEPLOY_EVIDENCE_DIR}/asr-release-${RELEASE_ID}.log"
release_status=0
run_with_evidence "${release_evidence}" \
  "${ROOT_DIR}/scripts/verify_asr_release.sh" release || release_status=$?
if ((release_status != 0)); then
  echo "Release verification failed with status ${release_status}; cutover was not attempted." >&2
  exit "${release_status}"
fi

assert_release_assets_unchanged
RELEASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' qwen-asr-api:latest)"
[[ -n "${RELEASE_IMAGE_ID}" ]] || fail "release verifier did not produce qwen-asr-api:latest"
docker tag "${RELEASE_IMAGE_ID}" "${RELEASE_IMAGE_REF}"
docker tag "${RELEASE_IMAGE_ID}" qwen-asr-api:latest

CUTOVER_STARTED=1
compose up -d --force-recreate --no-deps --no-build qwen-asr-api
assert_running_image "${RELEASE_IMAGE_ID}"
deploy_evidence="${ASR_DEPLOY_EVIDENCE_DIR}/asr-cutover-${RELEASE_ID}.log"
deploy_status=0
run_with_evidence "${deploy_evidence}" verify_local_deployment || deploy_status=$?
if ((deploy_status != 0)); then
  echo "Post-cutover readiness or smoke failed with status ${deploy_status}." >&2
  exit "${deploy_status}"
fi

if ((RUN_LIVE)); then
  live_evidence="${ASR_DEPLOY_EVIDENCE_DIR}/asr-live-${RELEASE_ID}.log"
  live_status=0
  run_with_evidence "${live_evidence}" \
    "${ROOT_DIR}/scripts/verify_asr_release.sh" live || live_status=$?
  if ((live_status != 0)); then
    echo "Live verification failed with status ${live_status}." >&2
    exit "${live_status}"
  fi
  LIVE_VERIFIED_IMAGE_ID="$(docker image inspect --format '{{.Id}}' qwen-asr-api:latest)"
  if [[ "${LIVE_VERIFIED_IMAGE_ID}" != "${RELEASE_IMAGE_ID}" ]]; then
    fail "live verification rebuilt a different image ID; refusing mixed-image evidence"
  fi
else
  echo "Live verification skipped by explicit operator request."
fi

# Live mode includes another release build and must reproduce the accepted ID.
# Keep the deployed and canonical tag pinned to that exact image.
docker tag "${RELEASE_IMAGE_ID}" qwen-asr-api:latest
assert_running_image "${RELEASE_IMAGE_ID}"
assert_release_assets_unchanged
DEPLOYMENT_COMPLETE=1

echo "ASR cloud deployment passed for commit ${CANDIDATE_SHA}."
echo "Release image: ${RELEASE_IMAGE_ID} (${RELEASE_IMAGE_REF})"
echo "Rollback image: ${ROLLBACK_IMAGE_ID} (${ROLLBACK_IMAGE_REF})"
echo "Protected evidence: ${ASR_DEPLOY_EVIDENCE_DIR}"
echo "Protected rollback backup: ${BACKUP_RUN_DIR}"
