#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${ASR_MONITOR_OUTPUT_ROOT:-/tmp/asr-monitor}"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MARKER_PATH="${OUTPUT_ROOT}/.asr-monitor-owned"
LOCK_DIR="${OUTPUT_ROOT}/.monitor.lock"
BASE_URL="${ASR_MONITOR_BASE_URL:-http://127.0.0.1:8002}"
SERVICE="${ASR_MONITOR_SERVICE:-qwen-asr-api}"
HYMT_SERVICE="${ASR_MONITOR_HYMT_SERVICE:-hy-mt-api}"
GPU_INDEX="${ASR_MONITOR_GPU_INDEX:-0}"
GPU_INTERVAL="${ASR_MONITOR_GPU_INTERVAL_SECONDS:-0.5}"
HTTP_INTERVAL="${ASR_MONITOR_HTTP_INTERVAL_SECONDS:-0.5}"
CONTAINER_INTERVAL="${ASR_MONITOR_CONTAINER_INTERVAL_SECONDS:-1}"
KEEP_RUNS="${ASR_MONITOR_KEEP_RUNS:-20}"
KEEP_DAYS="${ASR_MONITOR_KEEP_DAYS:-14}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
CURRENT_DIR="${RUNS_DIR}/${RUN_ID}"
ARCHIVE_PATH="${RUNS_DIR}/${RUN_ID}.tar.gz"
PARTIAL_ARCHIVE="${ARCHIVE_PATH}.partial"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ASR_CONTAINER_ID=""
HYMT_CONTAINER_ID=""
PIDS=()
FINALIZED=0

usage() {
  cat <<'EOF'
Usage:
  set -a
  source .env
  set +a
  scripts/monitor_asr_bottleneck.sh

The monitor runs until Ctrl+C. Every run creates a timestamped directory and
archive under /tmp/asr-monitor/runs, then generates report.json and report.md.

Optional environment variables:
  ASR_MONITOR_OUTPUT_ROOT
  ASR_MONITOR_BASE_URL
  ASR_MONITOR_SERVICE
  ASR_MONITOR_HYMT_SERVICE
  ASR_MONITOR_GPU_INDEX
  ASR_MONITOR_GPU_INTERVAL_SECONDS
  ASR_MONITOR_HTTP_INTERVAL_SECONDS
  ASR_MONITOR_CONTAINER_INTERVAL_SECONDS
  ASR_MONITOR_KEEP_RUNS
  ASR_MONITOR_KEEP_DAYS

API_KEY is read from the environment and is never intentionally written.
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
  printf '%s\t%s\t%s\n' "$(utc_now)" "$1" "$2" >>"${CURRENT_DIR}/collector-errors.log"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 2
  }
}

validate_nonnegative_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || {
    echo "$1 must be a nonnegative integer" >&2
    exit 2
  }
}

validate_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    echo "$1 must be a positive integer" >&2
    exit 2
  }
}

prepare_output() {
  if [[ "${OUTPUT_ROOT}" != /* || "${OUTPUT_ROOT}" == "/" || -L "${OUTPUT_ROOT}" ]]; then
    echo "ASR_MONITOR_OUTPUT_ROOT must be an absolute non-root nonsymlink path" >&2
    exit 2
  fi
  if [[ -d "${OUTPUT_ROOT}" && ! -e "${MARKER_PATH}" ]] &&
    find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing unmarked nonempty output root: ${OUTPUT_ROOT}" >&2
    exit 2
  fi
  mkdir -p -- "${RUNS_DIR}"
  : >"${MARKER_PATH}"
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another ASR monitor owns ${OUTPUT_ROOT}" >&2
    exit 2
  fi
  mkdir -- "${CURRENT_DIR}"
  : >"${CURRENT_DIR}/collector-errors.log"
}

write_metadata() {
  local image_id command_json asr_host_pid hymt_host_pid
  image_id="$(docker inspect --format '{{.Image}}' "${ASR_CONTAINER_ID}" 2>/dev/null || true)"
  command_json="$(docker inspect --format '{{json .Config.Cmd}}' "${ASR_CONTAINER_ID}" 2>/dev/null || true)"
  asr_host_pid="$(docker inspect --format '{{.State.Pid}}' "${ASR_CONTAINER_ID}" 2>/dev/null || true)"
  hymt_host_pid="$(docker inspect --format '{{.State.Pid}}' "${HYMT_CONTAINER_ID}" 2>/dev/null || true)"
  env \
    META_PATH="${CURRENT_DIR}/metadata.json" RUN_ID="${RUN_ID}" STARTED_UTC="${STARTED_UTC}" \
    SERVICE="${SERVICE}" HYMT_SERVICE="${HYMT_SERVICE}" BASE_URL="${BASE_URL}" \
    GPU_INDEX="${GPU_INDEX}" ASR_CONTAINER_ID="${ASR_CONTAINER_ID}" \
    HYMT_CONTAINER_ID="${HYMT_CONTAINER_ID}" IMAGE_ID="${image_id}" \
    ASR_HOST_PID="${asr_host_pid}" HYMT_HOST_PID="${hymt_host_pid}" \
    COMMAND_JSON="${command_json}" REPOSITORY_SHA="$(git rev-parse HEAD 2>/dev/null || true)" \
    python3 - <<'PY'
import json, os, platform
from pathlib import Path
payload = {
    "run_id": os.environ["RUN_ID"], "started_utc": os.environ["STARTED_UTC"],
    "hostname": platform.node(), "kernel": platform.platform(),
    "repository_sha": os.environ["REPOSITORY_SHA"], "service": os.environ["SERVICE"],
    "hymt_service": os.environ["HYMT_SERVICE"], "base_url": os.environ["BASE_URL"],
    "gpu_index": int(os.environ["GPU_INDEX"]), "asr_container_id": os.environ["ASR_CONTAINER_ID"],
    "hymt_container_id": os.environ["HYMT_CONTAINER_ID"], "image_id": os.environ["IMAGE_ID"],
    "asr_host_pid": os.environ["ASR_HOST_PID"], "hymt_host_pid": os.environ["HYMT_HOST_PID"],
    "container_command": os.environ["COMMAND_JSON"],
}
Path(os.environ["META_PATH"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_safe_config() {
  env CONFIG_PATH="${CURRENT_DIR}/config-safe.json" python3 - <<'PY'
import json, os
from pathlib import Path
names = (
    "ASR_BACKEND", "ASR_MODEL_NAME", "ASR_MODEL_ID", "ASR_DEVICE", "ASR_STREAM_MODE",
    "ASR_FASTER_WHISPER_BATCH_SIZE", "ASR_FASTER_WHISPER_PARTIAL_BEAM_SIZE",
    "ASR_FASTER_WHISPER_FINAL_BEAM_SIZE", "ASR_DIAGNOSTIC_LOGGING",
    "ASR_SLOW_ENGINE_LOG_SECONDS", "ASR_MAX_ACTIVE_STREAMS",
    "ASR_GATEWAY_MAX_ACTIVE_SESSIONS", "ASR_GATEWAY_DEFAULT_UPDATE_MS",
    "ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS", "ASR_GATEWAY_MAX_READY_JOBS",
    "ASR_GATEWAY_MAX_QUEUED_AUDIO_SECONDS", "ASR_GATEWAY_MAX_SESSION_BUFFER_SECONDS",
    "ASR_MAX_CONNECTION_LAG_SECONDS", "ASR_MAX_UNDECODED_AGE_SECONDS",
    "ASR_STREAM_QUEUE_TIMEOUT_SECONDS", "ASR_STREAM_INFERENCE_TIMEOUT_SECONDS",
    "ASR_MAX_UTTERANCE_SECONDS", "ASR_MAX_FRAME_BYTES", "ASR_WS_MAX_QUEUE",
)
payload = {name: os.environ[name] for name in names if name in os.environ}
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

gpu_collector() {
  local output="${CURRENT_DIR}/gpu.csv"
  echo 'sampled_at,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,power_watts,temperature_c,pstate,sm_clock_mhz,memory_clock_mhz' >"${output}"
  while true; do
    local value sampled_at
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate,clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s,%s\n' "${sampled_at}" "${value}" >>"${output}"
    else
      record_error gpu "nvidia-smi sample failed"
    fi
    sleep "${GPU_INTERVAL}"
  done
}

gpu_process_collector() {
  local output="${CURRENT_DIR}/gpu-processes.csv"
  echo 'sampled_at,pid,process_name,used_gpu_memory_mib' >"${output}"
  while true; do
    local value sampled_at
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)"; then
      while IFS= read -r line; do [[ -n "${line}" ]] && printf '%s,%s\n' "${sampled_at}" "${line}" >>"${output}"; done <<<"${value}"
    else
      record_error gpu-process "nvidia-smi process sample failed"
    fi
    sleep "${GPU_INTERVAL}"
  done
}

http_collector() {
  exec env ASR_MONITOR_HTTP_OUTPUT="${CURRENT_DIR}" ASR_MONITOR_HTTP_BASE_URL="${BASE_URL}" ASR_MONITOR_HTTP_INTERVAL="${HTTP_INTERVAL}" python3 - <<'PY'
import json, os, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
base = os.environ["ASR_MONITOR_HTTP_BASE_URL"].rstrip("/")
key = os.environ["API_KEY"]
interval = float(os.environ["ASR_MONITOR_HTTP_INTERVAL"])
output = Path(os.environ["ASR_MONITOR_HTTP_OUTPUT"])
targets = (("/ready", "readiness.jsonl", False), ("/v1/asr/metrics", "gateway-metrics.jsonl", True), ("/v1/asr/backends", "backends.jsonl", True))
while True:
    started = time.monotonic()
    sampled_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    for path, filename, authenticated in targets:
        record = {"sampled_at": sampled_at, "path": path}
        try:
            headers = {"X-API-Key": key} if authenticated else {}
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=3) as response:
                record.update(ok=True, status=response.status, response=json.load(response))
        except Exception as exc:
            record.update(ok=False, error_type=type(exc).__name__)
            with (output / "collector-errors.log").open("a") as errors:
                errors.write(f"{sampled_at}\thttp\t{path}: {type(exc).__name__}\n")
        with (output / filename).open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    time.sleep(max(0, interval - (time.monotonic() - started)))
PY
}

docker_stats_collector() {
  local output="${CURRENT_DIR}/docker-stats.csv"
  echo 'sampled_at,container,cpu_percent,memory_usage,memory_percent,network_io,block_io,pids' >"${output}"
  local containers=("${ASR_CONTAINER_ID}")
  [[ -n "${HYMT_CONTAINER_ID}" ]] && containers+=("${HYMT_CONTAINER_ID}")
  while true; do
    local value sampled_at
    sampled_at="$(utc_now)"
    if value="$(docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}|{{.NetIO}}|{{.BlockIO}}|{{.PIDs}}' "${containers[@]}" 2>/dev/null)"; then
      while IFS= read -r line; do [[ -n "${line}" ]] && printf '%s,%s\n' "${sampled_at}" "${line//|/,}" >>"${output}"; done <<<"${value}"
    else
      record_error docker-stats "docker stats sample failed"
    fi
    sleep "${CONTAINER_INTERVAL}"
  done
}

service_log_collector() {
  exec docker compose logs --follow --no-color --since "${STARTED_UTC}" "${SERVICE}" >>"${CURRENT_DIR}/asr-service.log" 2>>"${CURRENT_DIR}/collector-errors.log"
}

extract_events() {
  env LOG_PATH="${CURRENT_DIR}/asr-service.log" EVENT_PATH="${CURRENT_DIR}/events.jsonl" python3 - <<'PY'
import json, os
from pathlib import Path
source, destination = Path(os.environ["LOG_PATH"]), Path(os.environ["EVENT_PATH"])
with destination.open("w") as output:
    for line in source.read_text(errors="replace").splitlines():
        if "{" not in line:
            continue
        candidate = line[line.find("{"):]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == 1 and str(value.get("event", "")).startswith("asr_"):
            output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
PY
}

apply_retention() {
  env RUNS_DIR="${RUNS_DIR}" KEEP_RUNS="${KEEP_RUNS}" KEEP_DAYS="${KEEP_DAYS}" python3 - <<'PY'
import os, re, shutil, time
from pathlib import Path
runs = Path(os.environ["RUNS_DIR"])
keep, days = int(os.environ["KEEP_RUNS"]), int(os.environ["KEEP_DAYS"])
pattern = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
directories = sorted((p for p in runs.iterdir() if p.is_dir() and not p.is_symlink() and pattern.fullmatch(p.name) and (p / ".completed").is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
cutoff = time.time() - days * 86400
for index, path in enumerate(directories):
    if index >= keep or path.stat().st_mtime < cutoff:
        archive = runs / f"{path.name}.tar.gz"
        shutil.rmtree(path)
        if archive.is_file() and not archive.is_symlink():
            archive.unlink()
PY
}

start_collector() {
  "$@" &
  PIDS+=("$!")
}

finalize() {
  local requested_status="${1:-0}"
  ((FINALIZED)) && return
  FINALIZED=1
  trap - INT TERM EXIT
  local pid
  for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  extract_events
  python3 scripts/analyze_asr_bottleneck.py "${CURRENT_DIR}" >/dev/null || record_error analyzer "analysis failed"
  (cd "${CURRENT_DIR}" && find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum >manifest.sha256)
  : >"${CURRENT_DIR}/.completed"
  if grep -R -F -q -- "${API_KEY}" "${CURRENT_DIR}"; then
    echo "Refusing to archive: API_KEY found in evidence" >&2
    rm -rf -- "${LOCK_DIR}"
    exit 1
  fi
  rm -f -- "${PARTIAL_ARCHIVE}"
  tar -C "${RUNS_DIR}" -czf "${PARTIAL_ARCHIVE}" "${RUN_ID}" || {
    rm -rf -- "${LOCK_DIR}"
    exit 1
  }
  mv -- "${PARTIAL_ARCHIVE}" "${ARCHIVE_PATH}"
  apply_retention
  rm -rf -- "${LOCK_DIR}"
  echo
  echo "ASR bottleneck monitor stopped."
  echo "Run directory: ${CURRENT_DIR}"
  echo "Report: ${CURRENT_DIR}/report.md"
  echo "Archive to provide: ${ARCHIVE_PATH}"
  exit "${requested_status}"
}

require_command python3
require_command docker
require_command nvidia-smi
require_command tar
[[ -n "${API_KEY:-}" ]] || { echo "API_KEY must be exported" >&2; exit 2; }
validate_nonnegative_integer ASR_MONITOR_GPU_INDEX "${GPU_INDEX}"
validate_positive_integer ASR_MONITOR_KEEP_RUNS "${KEEP_RUNS}"
validate_positive_integer ASR_MONITOR_KEEP_DAYS "${KEEP_DAYS}"
prepare_output

ASR_CONTAINER_ID="$(docker compose ps -q "${SERVICE}" 2>/dev/null || true)"
HYMT_CONTAINER_ID="$(docker compose ps -q "${HYMT_SERVICE}" 2>/dev/null || true)"
if [[ -z "${ASR_CONTAINER_ID}" || "${ASR_CONTAINER_ID}" == *$'\n'* ]]; then
  echo "Expected exactly one running ASR container: ${SERVICE}" >&2
  rm -rf -- "${LOCK_DIR}"
  exit 2
fi
[[ -n "${HYMT_CONTAINER_ID}" ]] || record_error hymt "HY-MT container is not running"

write_metadata
write_safe_config
: >"${CURRENT_DIR}/asr-service.log"
: >"${CURRENT_DIR}/events.jsonl"
: >"${CURRENT_DIR}/gateway-metrics.jsonl"
: >"${CURRENT_DIR}/backends.jsonl"
: >"${CURRENT_DIR}/readiness.jsonl"

trap 'finalize 0' INT TERM
trap 'finalize $?' EXIT
start_collector gpu_collector
start_collector gpu_process_collector
start_collector http_collector
start_collector docker_stats_collector
start_collector service_log_collector

echo "ASR bottleneck monitor started."
echo "Run ID: ${RUN_ID}"
echo "Run directory: ${CURRENT_DIR}"
echo "Run the concurrency test in another terminal, then press Ctrl+C here."
while true; do sleep 1; done
