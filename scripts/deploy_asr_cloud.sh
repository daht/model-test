#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
RUN_LIVE=1
PYTHON_BIN=""
CANDIDATE_SHA=""
RELEASE_ID=""
RELEASE_IMAGE_ID=""
RELEASE_IMAGE_REF=""
ROLLBACK_IMAGE_ID=""
ROLLBACK_IMAGE_REF=""
ROLLBACK_CONTAINER_ID=""
BACKUP_RUN_DIR=""
ENV_BACKUP=""
MANIFEST_BACKUP=""
ROLLBACK_RECEIPT=""
RELEASE_RECEIPT=""
ENV_SHA256=""
MANIFEST_SHA256=""
MAINTENANCE_STARTED=0
DEPLOYMENT_COMPLETE=0
SNAPSHOT_READY=0

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_asr_cloud.sh [--dry-run] [--skip-live]

Verify, deploy, and live-verify one production ASR image during an explicit
single-GPU maintenance window. Full release/deploy/live verification is the
fail-closed default.

Options:
  --dry-run    Print the ordered workflow without checking prerequisites or changing anything.
  --skip-live  Stop after release verification, exact-image cutover, readiness, and smoke.
  -h, --help   Show this help.

Release environment (defaults shown):
  ASR_RELEASE_ENV_FILE       <repository>/.env
  ASR_RELEASE_MODEL_DIR      <repository>/models/Qwen3-ASR-1.7B
  ASR_RELEASE_MANIFEST       <repository>/models/Qwen3-ASR-1.7B.manifest.json

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
The existing ASR is stopped before release warmup, so this workflow has planned
downtime but never runs two Qwen model owners on one GPU.
EOF
}

print_plan() {
  cat <<'EOF'
DRY RUN: no prerequisites checked and no commands executed.
01 validate clean committed checkout and external release/live inputs
02 prepare repository .venv with pinned requirements-dev.txt when needed
03 snapshot current deployment image and back up .env plus approved manifest
04 verify and receipt the running rollback image/config/model/manifest baseline
05 stop the existing ASR model owner and begin the maintenance window
06 run scripts/verify_asr_release.sh release with protected evidence
07 receipt and tag the exact release-verified image ID
08 deploy the exact release-verified image without rebuilding
09 verify local readiness and WebSocket smoke
10 run evidence-bound deployed-live gates without another model owner
11 reassert the accepted image ID and unchanged config/model/manifest
On any failure or interruption after step 05: atomically restore the receipted
previous image, .env, and approved manifest, verify the unchanged model and
container baseline, then rerun local readiness and WebSocket smoke while
preserving the original exit status.
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
export ASR_RELEASE_MODEL_DIR="${ASR_RELEASE_MODEL_DIR:-${ROOT_DIR}/models/Qwen3-ASR-1.7B}"
export ASR_RELEASE_MANIFEST="${ASR_RELEASE_MANIFEST:-${ROOT_DIR}/models/Qwen3-ASR-1.7B.manifest.json}"
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
  require_regular_file "${ROOT_DIR}/scripts/asr_deploy_receipt.py"
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

assert_candidate_checkout_unchanged() {
  if [[ "$(git rev-parse HEAD)" != "${CANDIDATE_SHA}" ]]; then
    fail "candidate commit changed during deployment"
    return 1
  fi
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    fail "candidate checkout changed during deployment"
    return 1
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

assert_no_running_model_owner() {
  local ids
  local count
  ids="$(compose ps -q qwen-asr-api)"
  count="$(printf '%s\n' "${ids}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  if [[ "${count}" != "0" ]]; then
    fail "expected no running qwen-asr-api model owner during maintenance, found ${count}"
    return 1
  fi
}

stop_model_owner() {
  MAINTENANCE_STARTED=1
  compose stop qwen-asr-api
  assert_no_running_model_owner
}

snapshot_current_deployment() {
  local timestamp
  ROLLBACK_CONTAINER_ID="$(current_container_id)"
  ROLLBACK_IMAGE_ID="$(docker inspect --format '{{.Image}}' "${ROLLBACK_CONTAINER_ID}")"
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

validate_running_rollback_baseline() {
  if ! PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_deploy_receipt.py \
    validate-running-baseline \
    --container-id "${ROLLBACK_CONTAINER_ID}" \
    --repository "${ROOT_DIR}" \
    --env-file "${ASR_RELEASE_ENV_FILE}" \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"; then
    fail "running rollback baseline does not match the current image/config/model/manifest assets"
    return 1
  fi
}

verify_rollback_baseline() {
  assert_release_assets_unchanged || return $?
  validate_running_rollback_baseline || return $?
  verify_local_deployment || return $?
  echo "ASR rollback baseline verification passed."
}

create_deployment_receipt() {
  local kind="$1"
  local output="$2"
  local image_id="$3"
  local evidence="$4"
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_deploy_receipt.py \
    create-receipt \
    --kind "${kind}" \
    --output "${output}" \
    --repository "${ROOT_DIR}" \
    --candidate-sha "${CANDIDATE_SHA}" \
    --image-id "${image_id}" \
    --env-file "${ASR_RELEASE_ENV_FILE}" \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}" \
    --evidence "${evidence}"
}

validate_deployment_receipt() {
  local kind="$1"
  local receipt="$2"
  local image_id="$3"
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" scripts/asr_deploy_receipt.py \
    validate-receipt \
    --kind "${kind}" \
    --receipt "${receipt}" \
    --repository "${ROOT_DIR}" \
    --candidate-sha "${CANDIDATE_SHA}" \
    --image-id "${image_id}" \
    --env-file "${ASR_RELEASE_ENV_FILE}" \
    --model-dir "${ASR_RELEASE_MODEL_DIR}" \
    --manifest "${ASR_RELEASE_MANIFEST}"
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

  assert_candidate_checkout_unchanged || rollback_failed=1
  if [[ -L "${ASR_RELEASE_ENV_FILE}" || -L "${ASR_RELEASE_MANIFEST}" ]]; then
    echo "Rollback refused to write through a release asset symlink." >&2
    rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    compose stop qwen-asr-api || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    assert_no_running_model_owner || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    install -m 600 -- "${ENV_BACKUP}" "${ASR_RELEASE_ENV_FILE}" || rollback_failed=1
    install -m 600 -- "${MANIFEST_BACKUP}" "${ASR_RELEASE_MANIFEST}" || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    assert_release_assets_unchanged || rollback_failed=1
  fi
  if ((rollback_failed == 0)); then
    validate_deployment_receipt rollback-baseline "${ROLLBACK_RECEIPT}" "${ROLLBACK_IMAGE_ID}" \
      || rollback_failed=1
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
    ROLLBACK_CONTAINER_ID="$(current_container_id)"
    validate_running_rollback_baseline || rollback_failed=1
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
  if ((MAINTENANCE_STARTED && !DEPLOYMENT_COMPLETE)); then
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
  echo "Deployment interrupted; rollback will be attempted if maintenance started." >&2
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

baseline_evidence="${ASR_DEPLOY_EVIDENCE_DIR}/asr-rollback-baseline-${RELEASE_ID}.log"
baseline_status=0
run_with_evidence "${baseline_evidence}" verify_rollback_baseline || baseline_status=$?
if ((baseline_status != 0)); then
  echo "Rollback baseline verification failed with status ${baseline_status}; maintenance was not started." >&2
  exit "${baseline_status}"
fi
ROLLBACK_RECEIPT="${BACKUP_RUN_DIR}/rollback-baseline.receipt.json"
create_deployment_receipt rollback-baseline "${ROLLBACK_RECEIPT}" \
  "${ROLLBACK_IMAGE_ID}" "${baseline_evidence}"

echo "Entering the single-GPU ASR maintenance window; stopping the current model owner."
stop_model_owner

release_evidence="${ASR_DEPLOY_EVIDENCE_DIR}/asr-release-${RELEASE_ID}.log"
release_status=0
run_with_evidence "${release_evidence}" \
  "${ROOT_DIR}/scripts/verify_asr_release.sh" release || release_status=$?
if ((release_status != 0)); then
  echo "Release verification failed with status ${release_status}; cutover was not attempted." >&2
  exit "${release_status}"
fi

assert_no_running_model_owner
assert_release_assets_unchanged
RELEASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' qwen-asr-api:latest)"
[[ -n "${RELEASE_IMAGE_ID}" ]] || fail "release verifier did not produce qwen-asr-api:latest"
docker tag "${RELEASE_IMAGE_ID}" "${RELEASE_IMAGE_REF}"
docker tag "${RELEASE_IMAGE_ID}" qwen-asr-api:latest
RELEASE_RECEIPT="${BACKUP_RUN_DIR}/release.receipt.json"
create_deployment_receipt release "${RELEASE_RECEIPT}" \
  "${RELEASE_IMAGE_ID}" "${release_evidence}"
export ASR_DEPLOYED_RELEASE_RECEIPT="${RELEASE_RECEIPT}"

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
    "${ROOT_DIR}/scripts/verify_asr_release.sh" deployed-live || live_status=$?
  if ((live_status != 0)); then
    echo "Evidence-bound deployed-live verification failed with status ${live_status}." >&2
    exit "${live_status}"
  fi
else
  echo "Live verification skipped by explicit operator request."
fi

# Keep the deployed and canonical tag pinned to the exact release image.
docker tag "${RELEASE_IMAGE_ID}" qwen-asr-api:latest
assert_running_image "${RELEASE_IMAGE_ID}"
assert_candidate_checkout_unchanged
assert_release_assets_unchanged
DEPLOYMENT_COMPLETE=1

echo "ASR cloud deployment passed for commit ${CANDIDATE_SHA}."
echo "Release image: ${RELEASE_IMAGE_ID} (${RELEASE_IMAGE_REF})"
echo "Rollback image: ${ROLLBACK_IMAGE_ID} (${ROLLBACK_IMAGE_REF})"
echo "Protected evidence: ${ASR_DEPLOY_EVIDENCE_DIR}"
echo "Protected rollback backup: ${BACKUP_RUN_DIR}"
