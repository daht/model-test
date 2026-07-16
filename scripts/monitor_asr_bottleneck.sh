#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${ASR_MONITOR_OUTPUT_ROOT:-/tmp/asr-bottleneck-monitor}"
CURRENT_DIR="${OUTPUT_ROOT}/current"
ARCHIVE_PATH="${OUTPUT_ROOT}/asr-bottleneck-current.tar.gz"
MARKER_PATH="${OUTPUT_ROOT}/.asr-bottleneck-monitor-owned"
BASE_URL="${ASR_MONITOR_BASE_URL:-http://127.0.0.1:8002}"
SERVICE="${ASR_MONITOR_SERVICE:-qwen-asr-api}"
GPU_INDEX="${ASR_MONITOR_GPU_INDEX:-0}"
GPU_INTERVAL="${ASR_MONITOR_GPU_INTERVAL_SECONDS:-1}"
HTTP_INTERVAL="${ASR_MONITOR_HTTP_INTERVAL_SECONDS:-0.5}"
CONTAINER_INTERVAL="${ASR_MONITOR_CONTAINER_INTERVAL_SECONDS:-1}"

PIDS=()
FINALIZED=0
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CONTAINER_ID=""

usage() {
  cat <<'EOF'
Usage:
  set -a
  source .env
  set +a
  scripts/monitor_asr_bottleneck.sh

The monitor runs until Ctrl+C and writes:
  /tmp/asr-bottleneck-monitor/asr-bottleneck-current.tar.gz

Optional environment variables:
  ASR_MONITOR_OUTPUT_ROOT
  ASR_MONITOR_BASE_URL
  ASR_MONITOR_SERVICE
  ASR_MONITOR_GPU_INDEX
  ASR_MONITOR_GPU_INTERVAL_SECONDS
  ASR_MONITOR_HTTP_INTERVAL_SECONDS
  ASR_MONITOR_CONTAINER_INTERVAL_SECONDS

Every run deletes only the previous managed current/ directory and archive.
API_KEY is read from the environment and is never written intentionally.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ
}

record_error() {
  local source="$1"
  local message="$2"
  printf '%s\t%s\t%s\n' "$(utc_now)" "${source}" "${message}" \
    >>"${CURRENT_DIR}/collector-errors.log"
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing required command: ${command_name}" >&2
    exit 2
  fi
}

validate_output_root() {
  if [[ "${OUTPUT_ROOT}" != /* || "${OUTPUT_ROOT}" == "/" ]]; then
    echo "ASR_MONITOR_OUTPUT_ROOT must be an absolute non-root path" >&2
    exit 2
  fi
  if [[ -L "${OUTPUT_ROOT}" ]]; then
    echo "ASR_MONITOR_OUTPUT_ROOT must not be a symbolic link" >&2
    exit 2
  fi
  if [[ -d "${OUTPUT_ROOT}" && ! -e "${MARKER_PATH}" ]]; then
    if find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "Refusing to clean nonempty unmarked output directory: ${OUTPUT_ROOT}" >&2
      exit 2
    fi
  fi
  mkdir -p -- "${OUTPUT_ROOT}"
  : >"${MARKER_PATH}"

  # Clean only paths owned by this monitor. Never use a broad /tmp glob.
  rm -rf -- "${CURRENT_DIR}"
  rm -f -- "${ARCHIVE_PATH}"
  mkdir -p -- "${CURRENT_DIR}"
  : >"${CURRENT_DIR}/collector-errors.log"
}

write_metadata() {
  local image_id=""
  local command_json=""
  image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  command_json="$(docker inspect --format '{{json .Config.Cmd}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  {
    printf 'started_utc=%s\n' "${STARTED_UTC}"
    printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
    printf 'kernel=%s\n' "$(uname -a 2>/dev/null || true)"
    printf 'repository_sha=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
    printf 'service=%s\n' "${SERVICE}"
    printf 'base_url=%s\n' "${BASE_URL}"
    printf 'gpu_index=%s\n' "${GPU_INDEX}"
    printf 'container_id=%s\n' "${CONTAINER_ID}"
    printf 'image_id=%s\n' "${image_id}"
    printf 'container_command=%s\n' "${command_json}"
  } >"${CURRENT_DIR}/metadata.txt"
}

write_safe_config() {
  local names=(
    ASR_BACKEND ASR_MODEL_NAME ASR_MODEL_ID ASR_DEVICE ASR_TORCH_DTYPE
    ASR_STREAM_MODE ASR_STREAM_CHUNK_SECONDS ASR_MAX_ACTIVE_STREAMS
    ASR_GATEWAY_MAX_ACTIVE_SESSIONS ASR_GATEWAY_DEFAULT_UPDATE_MS
    ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS ASR_GATEWAY_MAX_READY_JOBS
    ASR_GATEWAY_MAX_QUEUED_AUDIO_SECONDS ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS
    ASR_MAX_CONNECTION_LAG_SECONDS ASR_MAX_UNDECODED_AGE_SECONDS
    ASR_STREAM_QUEUE_TIMEOUT_SECONDS ASR_STREAM_INFERENCE_TIMEOUT_SECONDS
    ASR_VLLM_GPU_MEMORY_UTILIZATION ASR_VLLM_MAX_MODEL_LEN
    ASR_VLLM_MAX_NEW_TOKENS ASR_VAD_MIN_SPEECH_MS ASR_VAD_MIN_SILENCE_MS
    ASR_VAD_HANGOVER_MS ASR_VAD_PRE_ROLL_MS ASR_MAX_UTTERANCE_SECONDS
    ASR_MAX_FRAME_BYTES ASR_WS_MAX_QUEUE
  )
  : >"${CURRENT_DIR}/config-safe.txt"
  local name
  for name in "${names[@]}"; do
    if [[ -n "${!name+x}" ]]; then
      printf '%s=%s\n' "${name}" "${!name}" >>"${CURRENT_DIR}/config-safe.txt"
    fi
  done
}

gpu_collector() {
  local output="${CURRENT_DIR}/gpu.csv"
  printf '%s\n' \
    'sampled_at,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,power_watts,temperature_c,pstate,sm_clock_mhz,memory_clock_mhz' \
    >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --id="${GPU_INDEX}" \
      --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate,clocks.sm,clocks.mem \
      --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s,%s\n' "${sampled_at}" "${value}" >>"${output}"
    else
      record_error gpu "nvidia-smi sample failed"
    fi
    sleep "${GPU_INTERVAL}"
  done
}

http_collector() {
  exec env \
    ASR_MONITOR_HTTP_OUTPUT="${CURRENT_DIR}" \
    ASR_MONITOR_HTTP_BASE_URL="${BASE_URL}" \
    ASR_MONITOR_HTTP_INTERVAL="${HTTP_INTERVAL}" \
    python3 - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

base = os.environ["ASR_MONITOR_HTTP_BASE_URL"].rstrip("/")
key = os.environ["API_KEY"]
interval = float(os.environ["ASR_MONITOR_HTTP_INTERVAL"])
output = Path(os.environ["ASR_MONITOR_HTTP_OUTPUT"])
targets = (
    ("/ready", output / "readiness.jsonl", False),
    ("/v1/asr/metrics", output / "gateway-metrics.jsonl", True),
    ("/v1/asr/backends", output / "backends.jsonl", True),
)

while True:
    started = time.monotonic()
    sampled_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    for path, destination, authenticated in targets:
        headers = {"X-API-Key": key} if authenticated else {}
        request = urllib.request.Request(base + path, headers=headers)
        record = {"sampled_at": sampled_at, "path": path}
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                record["status"] = response.status
                record["response"] = json.load(response)
                record["ok"] = True
        except Exception as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            with (output / "collector-errors.log").open("a") as errors:
                errors.write(f"{sampled_at}\thttp\t{path}: {type(exc).__name__}\n")
        with destination.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    delay = interval - (time.monotonic() - started)
    if delay > 0:
        time.sleep(delay)
PY
}

docker_stats_collector() {
  local output="${CURRENT_DIR}/docker-stats.csv"
  printf '%s\n' 'sampled_at,container,cpu_percent,memory_usage,memory_percent,network_io,block_io,pids' >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(docker stats --no-stream \
      --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}|{{.NetIO}}|{{.BlockIO}}|{{.PIDs}}' \
      "${CONTAINER_ID}" 2>/dev/null)"; then
      printf '%s,%s\n' "${sampled_at}" "${value//|/,}" >>"${output}"
    else
      record_error docker-stats "docker stats sample failed"
    fi
    sleep "${CONTAINER_INTERVAL}"
  done
}

service_log_collector() {
  exec docker compose logs --follow --no-color --since "${STARTED_UTC}" "${SERVICE}" \
    >>"${CURRENT_DIR}/asr-service.log" 2>>"${CURRENT_DIR}/collector-errors.log"
}

start_collector() {
  "$@" &
  PIDS+=("$!")
}

finalize() {
  local requested_status="${1:-0}"
  if ((FINALIZED)); then
    return
  fi
  FINALIZED=1
  trap - INT TERM EXIT

  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done

  local finished_utc
  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'finished_utc=%s\n' "${finished_utc}" >>"${CURRENT_DIR}/metadata.txt"
  {
    printf 'started_utc=%s\n' "${STARTED_UTC}"
    printf 'finished_utc=%s\n' "${finished_utc}"
    printf 'gpu_samples=%s\n' "$(( $(wc -l <"${CURRENT_DIR}/gpu.csv" 2>/dev/null || echo 1) - 1 ))"
    printf 'metrics_samples=%s\n' "$(wc -l <"${CURRENT_DIR}/gateway-metrics.jsonl" 2>/dev/null || echo 0)"
    printf 'backend_samples=%s\n' "$(wc -l <"${CURRENT_DIR}/backends.jsonl" 2>/dev/null || echo 0)"
    printf 'readiness_samples=%s\n' "$(wc -l <"${CURRENT_DIR}/readiness.jsonl" 2>/dev/null || echo 0)"
    printf 'collector_errors=%s\n' "$(wc -l <"${CURRENT_DIR}/collector-errors.log" 2>/dev/null || echo 0)"
  } >"${CURRENT_DIR}/summary.txt"

  if grep -R -F -q -- "${API_KEY}" "${CURRENT_DIR}"; then
    echo "Refusing to archive: API_KEY was found in collected evidence" >&2
    exit 1
  fi

  if ! tar -C "${OUTPUT_ROOT}" -czf "${ARCHIVE_PATH}" current; then
    echo "Unable to create monitor archive" >&2
    exit 1
  fi
  echo
  echo "ASR bottleneck monitor stopped."
  echo "Evidence directory: ${CURRENT_DIR}"
  echo "Archive to provide: ${ARCHIVE_PATH}"
  exit "${requested_status}"
}

require_command python3
require_command docker
require_command nvidia-smi
require_command tar

if [[ -z "${API_KEY:-}" ]]; then
  echo "API_KEY must be exported in the environment" >&2
  exit 2
fi
if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ASR_MONITOR_GPU_INDEX must be a nonnegative integer" >&2
  exit 2
fi

validate_output_root

CONTAINER_ID="$(docker compose ps -q "${SERVICE}" 2>/dev/null || true)"
if [[ -z "${CONTAINER_ID}" || "${CONTAINER_ID}" == *$'\n'* ]]; then
  echo "Expected exactly one running container for service: ${SERVICE}" >&2
  exit 2
fi
if ! nvidia-smi --id="${GPU_INDEX}" \
  --query-gpu=index \
  --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU index ${GPU_INDEX} is unavailable" >&2
  exit 2
fi

write_metadata
write_safe_config
: >"${CURRENT_DIR}/asr-service.log"
: >"${CURRENT_DIR}/gateway-metrics.jsonl"
: >"${CURRENT_DIR}/backends.jsonl"
: >"${CURRENT_DIR}/readiness.jsonl"

trap 'finalize 0' INT TERM
trap 'finalize $?' EXIT

start_collector gpu_collector
start_collector http_collector
start_collector docker_stats_collector
start_collector service_log_collector

echo "ASR bottleneck monitor started."
echo "Service: ${SERVICE}"
echo "GPU index: ${GPU_INDEX}"
echo "Evidence directory: ${CURRENT_DIR}"
echo "Run the concurrency test in another terminal."
echo "Press Ctrl+C here after the test finishes."

while true; do
  sleep 1
done
