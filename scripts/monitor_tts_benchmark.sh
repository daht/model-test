#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${TTS_MONITOR_OUTPUT_ROOT:-/tmp/tts-monitor}"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MARKER_PATH="${OUTPUT_ROOT}/.tts-monitor-owned"
LOCK_DIR="${OUTPUT_ROOT}/.monitor.lock"
SERVICE="${TTS_MONITOR_SERVICE:-cosyvoice-tts-api}"
GPU_INDEX="${TTS_MONITOR_GPU_INDEX:-0}"
GPU_INTERVAL="${TTS_MONITOR_GPU_INTERVAL_SECONDS:-0.5}"
CONTAINER_INTERVAL="${TTS_MONITOR_CONTAINER_INTERVAL_SECONDS:-1}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
CURRENT_DIR="${RUNS_DIR}/${RUN_ID}"
ARCHIVE_PATH="${RUNS_DIR}/${RUN_ID}.tar.gz"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
CONTAINER_ID=""
PIDS=()
FINALIZED=0

usage() {
  cat <<'EOF'
Usage: scripts/monitor_tts_benchmark.sh

Collects TTS service logs, GPU samples, GPU processes, and container stats.
Start this on the A10 server, run the remote benchmark, then press Ctrl+C.

Optional environment variables:
  TTS_MONITOR_OUTPUT_ROOT
  TTS_MONITOR_SERVICE
  TTS_MONITOR_GPU_INDEX
  TTS_MONITOR_GPU_INTERVAL_SECONDS
  TTS_MONITOR_CONTAINER_INTERVAL_SECONDS

The collector never sends TTS requests and does not require API_KEY.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || { usage; exit 0; }

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 2
  }
}

validate_positive_decimal() {
  python3 - "$1" "$2" <<'PY'
import math, sys
try:
    value = float(sys.argv[2])
except ValueError:
    raise SystemExit(f"{sys.argv[1]} must be a positive number")
if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"{sys.argv[1]} must be a positive number")
PY
}

prepare_output() {
  if [[ "${OUTPUT_ROOT}" != /* || "${OUTPUT_ROOT}" == "/" || -L "${OUTPUT_ROOT}" ]]; then
    echo "TTS_MONITOR_OUTPUT_ROOT must be an absolute non-root nonsymlink path" >&2
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
    echo "Another TTS monitor owns ${OUTPUT_ROOT}" >&2
    exit 2
  fi
  mkdir -- "${CURRENT_DIR}"
  : >"${CURRENT_DIR}/collector-errors.log"
}

release_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

resolve_container() {
  local raw
  raw="$(docker compose ps -q "${SERVICE}" 2>/dev/null || true)"
  mapfile -t container_ids < <(printf '%s\n' "${raw}" | sed '/^[[:space:]]*$/d')
  if [[ ${#container_ids[@]} -ne 1 ]]; then
    echo "Expected exactly one running TTS container: ${SERVICE}" >&2
    release_lock
    exit 2
  fi
  CONTAINER_ID="${container_ids[0]}"
}

record_error() {
  printf '%s\t%s\t%s\n' "$(utc_now)" "$1" "$2" >>"${CURRENT_DIR}/collector-errors.log"
}

write_metadata() {
  local finished_utc="${1:-}" image_id started_at running restart_count oom_killed gpu_name
  image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  started_at="$(docker inspect --format '{{.State.StartedAt}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  running="$(docker inspect --format '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  restart_count="$(docker inspect --format '{{.RestartCount}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  oom_killed="$(docker inspect --format '{{.State.OOMKilled}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  gpu_name="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name --format=csv,noheader,nounits 2>/dev/null || true)"
  env META_PATH="${CURRENT_DIR}/metadata.json" RUN_ID="${RUN_ID}" \
    STARTED_UTC="${STARTED_UTC}" FINISHED_UTC="${finished_utc}" \
    HOSTNAME_VALUE="$(hostname)" KERNEL_VALUE="$(uname -a)" \
    REPOSITORY_SHA="$(git rev-parse HEAD 2>/dev/null || true)" SERVICE="${SERVICE}" \
    CONTAINER_ID="${CONTAINER_ID}" IMAGE_ID="${image_id}" CONTAINER_STARTED_AT="${started_at}" \
    CONTAINER_RUNNING="${running}" RESTART_COUNT="${restart_count}" OOM_KILLED="${oom_killed}" \
    GPU_INDEX="${GPU_INDEX}" GPU_NAME="${gpu_name}" python3 - <<'PY'
import json, os
from pathlib import Path

def integer(value):
    try:
        return int(value)
    except ValueError:
        return None

payload = {
    "run_id": os.environ["RUN_ID"],
    "started_utc": os.environ["STARTED_UTC"],
    "finished_utc": os.environ["FINISHED_UTC"] or None,
    "hostname": os.environ["HOSTNAME_VALUE"],
    "kernel": os.environ["KERNEL_VALUE"],
    "repository_sha": os.environ["REPOSITORY_SHA"],
    "service": os.environ["SERVICE"],
    "container_id": os.environ["CONTAINER_ID"],
    "image_id": os.environ["IMAGE_ID"],
    "container_started_at": os.environ["CONTAINER_STARTED_AT"],
    "container_running": os.environ["CONTAINER_RUNNING"].lower() == "true",
    "restart_count": integer(os.environ["RESTART_COUNT"]),
    "oom_killed": os.environ["OOM_KILLED"].lower() == "true",
    "gpu_index": integer(os.environ["GPU_INDEX"]),
    "gpu_name": os.environ["GPU_NAME"],
}
Path(os.environ["META_PATH"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

gpu_collector() {
  local output="${CURRENT_DIR}/gpu.csv"
  echo 'sampled_at,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,power_watts,temperature_c,pstate,sm_clock_mhz,memory_clock_mhz' >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate,clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null)" && [[ -n "${value}" ]]; then
      printf '%s,%s\n' "${sampled_at}" "${value}" >>"${output}"
    else
      record_error gpu sample_failed
    fi
    sleep "${GPU_INTERVAL}"
  done
}

gpu_process_collector() {
  local output="${CURRENT_DIR}/gpu-processes.csv"
  echo 'sampled_at,pid,process_name,used_gpu_memory_mib' >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)"; then
      while IFS= read -r line; do
        [[ -n "${line}" ]] && printf '%s,%s\n' "${sampled_at}" "${line}" >>"${output}"
      done <<<"${value}"
    else
      record_error gpu_process sample_failed
    fi
    sleep "${GPU_INTERVAL}"
  done
}

container_collector() {
  local output="${CURRENT_DIR}/container.csv"
  echo 'sampled_at|cpu_percent|memory_usage|memory_percent|network_io|block_io|pids' >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(docker stats --no-stream --format '{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}|{{.NetIO}}|{{.BlockIO}}|{{.PIDs}}' "${CONTAINER_ID}" 2>/dev/null)"; then
      printf '%s|%s\n' "${sampled_at}" "${value}" >>"${output}"
    else
      record_error container sample_failed
    fi
    sleep "${CONTAINER_INTERVAL}"
  done
}

service_log_collector() {
  exec docker compose logs --follow --timestamps --since "${STARTED_UTC}" "${SERVICE}" \
    >"${CURRENT_DIR}/service.log" 2>&1
}

start_collector() {
  "$@" &
  PIDS+=("$!")
}

stop_collectors() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
  PIDS=()
}

generate_report() {
  env RUN_DIR="${CURRENT_DIR}" python3 - <<'PY'
import csv, json, math, re
from pathlib import Path
import os

run = Path(os.environ["RUN_DIR"])

def number(value):
    try:
        result = float(str(value).strip().rstrip("%"))
    except ValueError:
        return None
    return result if math.isfinite(result) else None

def metric(values):
    clean = [value for value in values if value is not None]
    return {
        "average": sum(clean) / len(clean) if clean else None,
        "maximum": max(clean) if clean else None,
    }

gpu_rows = list(csv.DictReader((run / "gpu.csv").open()))
gpu = {
    "sample_count": len(gpu_rows),
    "utilization_percent": metric([number(row["gpu_util_percent"]) for row in gpu_rows]),
    "memory_used_mib": metric([number(row["memory_used_mib"]) for row in gpu_rows]),
    "power_watts": metric([number(row["power_watts"]) for row in gpu_rows]),
    "temperature_c": metric([number(row["temperature_c"]) for row in gpu_rows]),
}

container_rows = []
with (run / "container.csv").open() as source:
    header = source.readline().strip().split("|")
    for line in source:
        values = line.strip().split("|")
        if len(values) == len(header):
            container_rows.append(dict(zip(header, values)))
container = {
    "sample_count": len(container_rows),
    "cpu_percent": metric([number(row["cpu_percent"]) for row in container_rows]),
    "memory_percent": metric([number(row["memory_percent"]) for row in container_rows]),
    "pids": metric([number(row["pids"]) for row in container_rows]),
}

errors = [line for line in (run / "collector-errors.log").read_text().splitlines() if line]
report = {"gpu": gpu, "container": container, "collector_errors": errors}
(run / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

def show(value):
    return "-" if value is None else f"{value:.2f}"

lines = [
    "# TTS benchmark server monitor",
    "",
    f"- GPU samples: {gpu['sample_count']}",
    f"- GPU utilization avg/max: {show(gpu['utilization_percent']['average'])}/{show(gpu['utilization_percent']['maximum'])}%",
    f"- GPU memory avg/max: {show(gpu['memory_used_mib']['average'])}/{show(gpu['memory_used_mib']['maximum'])} MiB",
    f"- GPU power avg/max: {show(gpu['power_watts']['average'])}/{show(gpu['power_watts']['maximum'])} W",
    f"- Container CPU avg/max: {show(container['cpu_percent']['average'])}/{show(container['cpu_percent']['maximum'])}%",
    f"- Collector errors: {len(errors)}",
    "",
]
(run / "report.md").write_text("\n".join(lines))
PY
}

finalize() {
  [[ ${FINALIZED} -eq 0 ]] || return 0
  FINALIZED=1
  trap - INT TERM EXIT
  stop_collectors
  local finished_utc
  finished_utc="$(utc_now)"
  write_metadata "${finished_utc}"
  sed -E 's/([0-9]{1,3}\.){3}[0-9]{1,3}/[redacted-ip]/g' \
    "${CURRENT_DIR}/service.log" >"${CURRENT_DIR}/service.log.sanitized" || true
  [[ -e "${CURRENT_DIR}/service.log.sanitized" ]] && \
    mv -- "${CURRENT_DIR}/service.log.sanitized" "${CURRENT_DIR}/service.log"
  generate_report
  (
    cd "${CURRENT_DIR}"
    find . -maxdepth 1 -type f ! -name manifest.sha256 -printf '%P\0' | sort -z | \
      xargs -0 sha256sum >manifest.sha256
  )
  : >"${CURRENT_DIR}/.completed"
  tar -czf "${ARCHIVE_PATH}.partial" -C "${RUNS_DIR}" "${RUN_ID}"
  mv -- "${ARCHIVE_PATH}.partial" "${ARCHIVE_PATH}"
  release_lock
  echo "TTS monitor archive: ${ARCHIVE_PATH}"
}

handle_signal() {
  finalize
  exit 0
}

for command in docker nvidia-smi python3 tar sha256sum; do require_command "${command}"; done
validate_positive_decimal TTS_MONITOR_GPU_INTERVAL_SECONDS "${GPU_INTERVAL}"
validate_positive_decimal TTS_MONITOR_CONTAINER_INTERVAL_SECONDS "${CONTAINER_INTERVAL}"
prepare_output
resolve_container
write_metadata ""
trap handle_signal INT TERM
trap finalize EXIT
start_collector gpu_collector
start_collector gpu_process_collector
start_collector container_collector
start_collector service_log_collector

echo "TTS benchmark monitor started."
echo "Evidence directory: ${CURRENT_DIR}"
echo "Run the benchmark, then press Ctrl+C."
while true; do sleep 1; done
