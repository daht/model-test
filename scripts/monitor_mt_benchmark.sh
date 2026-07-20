#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${MT_MONITOR_OUTPUT_ROOT:-/tmp/mt-monitor}"
RUNS_DIR="${OUTPUT_ROOT}/runs"
MARKER_PATH="${OUTPUT_ROOT}/.mt-monitor-owned"
LOCK_DIR="${OUTPUT_ROOT}/.monitor.lock"
SERVICE="${MT_MONITOR_SERVICE:-hy-mt-api}"
GPU_INDEX="${MT_MONITOR_GPU_INDEX:-0}"
GPU_INTERVAL="${MT_MONITOR_GPU_INTERVAL_SECONDS:-0.5}"
CONTAINER_INTERVAL="${MT_MONITOR_CONTAINER_INTERVAL_SECONDS:-1}"
KEEP_RUNS="${MT_MONITOR_KEEP_RUNS:-20}"
KEEP_DAYS="${MT_MONITOR_KEEP_DAYS:-14}"

RUN_ID=""
CURRENT_DIR=""
ARCHIVE_PATH=""
PARTIAL_ARCHIVE=""
STARTED_UTC=""
CONTAINER_ID=""
PIDS=()
FINALIZED=0

usage() {
  cat <<'EOF'
Usage: scripts/monitor_mt_benchmark.sh

Starts a monitor-only evidence collector for the HY-MT Docker Compose service.
Run the MT benchmark from another machine, then press Ctrl+C here.

Optional environment variables:
  MT_MONITOR_SERVICE
  MT_MONITOR_GPU_INDEX
  MT_MONITOR_OUTPUT_ROOT
  MT_MONITOR_GPU_INTERVAL_SECONDS
  MT_MONITOR_CONTAINER_INTERVAL_SECONDS
  MT_MONITOR_KEEP_RUNS
  MT_MONITOR_KEEP_DAYS
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

validate_nonnegative_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || { echo "$1 must be a nonnegative integer" >&2; exit 2; }
}

validate_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$1 must be a positive integer" >&2; exit 2; }
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
  [[ $? -eq 0 ]] || exit 2
}

prepare_output() {
  if [[ "${OUTPUT_ROOT}" != /* || "${OUTPUT_ROOT}" == "/" || -L "${OUTPUT_ROOT}" ]]; then
    echo "MT_MONITOR_OUTPUT_ROOT must be an absolute non-root nonsymlink path" >&2
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
    echo "Another MT monitor owns ${OUTPUT_ROOT}" >&2
    exit 2
  fi
}

release_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

resolve_container() {
  local raw
  raw="$(docker compose ps -q "${SERVICE}" 2>/dev/null || true)"
  mapfile -t container_ids < <(printf '%s\n' "${raw}" | sed '/^[[:space:]]*$/d')
  if [[ ${#container_ids[@]} -ne 1 ]]; then
    echo "Expected exactly one running MT container: ${SERVICE}" >&2
    release_lock
    exit 2
  fi
  CONTAINER_ID="${container_ids[0]}"
}

record_error() {
  [[ -n "${CURRENT_DIR}" ]] || return 0
  printf '%s\t%s\t%s\n' "$(utc_now)" "$1" "$2" >>"${CURRENT_DIR}/collector-errors.log"
}

write_metadata() {
  local image_id started_at gpu_name running restart_count oom_killed
  image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  started_at="$(docker inspect --format '{{.State.StartedAt}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  running="$(docker inspect --format '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  restart_count="$(docker inspect --format '{{.RestartCount}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  oom_killed="$(docker inspect --format '{{.State.OOMKilled}}' "${CONTAINER_ID}" 2>/dev/null || true)"
  gpu_name="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name --format=csv,noheader,nounits 2>/dev/null || true)"
  env META_PATH="${CURRENT_DIR}/metadata.json" RUN_ID="${RUN_ID}" STARTED_UTC="${STARTED_UTC}" \
    FINISHED_UTC="${1:-}" HOSTNAME_VALUE="$(hostname)" KERNEL_VALUE="$(uname -a)" \
    REPOSITORY_SHA="$(git rev-parse HEAD 2>/dev/null || true)" SERVICE="${SERVICE}" \
    CONTAINER_ID="${CONTAINER_ID}" IMAGE_ID="${image_id}" CONTAINER_STARTED_AT="${started_at}" \
    CONTAINER_RUNNING="${running}" CONTAINER_RESTART_COUNT="${restart_count}" \
    CONTAINER_OOM_KILLED="${oom_killed}" GPU_INDEX="${GPU_INDEX}" GPU_NAME="${gpu_name}" \
    python3 - <<'PY'
import json, os
from pathlib import Path

def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
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
    "container_restart_count": integer(os.environ["CONTAINER_RESTART_COUNT"]),
    "container_oom_killed": os.environ["CONTAINER_OOM_KILLED"].lower() == "true",
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
  exec docker compose logs --follow --timestamps --since "${STARTED_UTC}" "${SERVICE}" >"${CURRENT_DIR}/service.log" 2>&1
}

start_collector() {
  "$1" &
  PIDS+=("$!")
}

stop_collectors() {
  local pid
  for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
  PIDS=()
}

generate_reports() {
  env RUN_DIR="${CURRENT_DIR}" python3 - <<'PY'
import csv, json, math, re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

run = Path(__import__("os").environ["RUN_DIR"])
warnings = []

def number(value):
    try:
        result = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]

def metric(values):
    clean = [value for value in values if value is not None]
    return {
        "average": sum(clean) / len(clean) if clean else None,
        "p95": nearest_rank(clean, 95),
        "maximum": max(clean) if clean else None,
    }

units = {"B": 1, "kB": 1000, "KiB": 1024, "MB": 1000**2, "MiB": 1024**2, "GB": 1000**3, "GiB": 1024**3}
def bytes_value(value):
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]+)\s*", value)
    if not match or match.group(2) not in units:
        return None
    return float(match.group(1)) * units[match.group(2)]

gpu_rows = list(csv.DictReader((run / "gpu.csv").open()))
gpu = {
    "sample_count": len(gpu_rows),
    "utilization_percent": metric([number(row["gpu_util_percent"]) for row in gpu_rows]),
    "memory_utilization_percent": metric([number(row["memory_util_percent"]) for row in gpu_rows]),
    "memory_used_mib": metric([number(row["memory_used_mib"]) for row in gpu_rows]),
    "power_watts": metric([number(row["power_watts"]) for row in gpu_rows]),
    "temperature_c": metric([number(row["temperature_c"]) for row in gpu_rows]),
}
if not gpu_rows: warnings.append("missing GPU samples")

with (run / "container.csv").open() as handle:
    container_rows = list(csv.DictReader(handle, delimiter="|"))
memory_values = []
for row in container_rows:
    usage = row["memory_usage"].split("/", 1)[0].strip()
    parsed = bytes_value(usage)
    if parsed is None: warnings.append("unparseable container memory unit")
    memory_values.append(parsed)
container = {
    "sample_count": len(container_rows),
    "cpu_percent": metric([number(row["cpu_percent"]) for row in container_rows]),
    "memory_used_bytes": metric(memory_values),
    "memory_percent": metric([number(row["memory_percent"]) for row in container_rows]),
}
if not container_rows: warnings.append("missing container samples")

process_rows = list(csv.DictReader((run / "gpu-processes.csv").open()))
process_peaks = defaultdict(list)
for row in process_rows:
    process_peaks[f"{row['pid'].strip()}:{row['process_name'].strip()}"].append(number(row["used_gpu_memory_mib"]))
gpu_processes = {key: max(value for value in values if value is not None) for key, values in process_peaks.items() if any(value is not None for value in values)}

error_lines = [line for line in (run / "collector-errors.log").read_text().splitlines() if line]
categories = Counter(line.split("\t")[1] if "\t" in line else "unknown" for line in error_lines)
metadata = json.loads((run / "metadata.json").read_text())
duration = None
if metadata.get("started_utc") and metadata.get("finished_utc"):
    start = datetime.fromisoformat(metadata["started_utc"].replace("Z", "+00:00"))
    finish = datetime.fromisoformat(metadata["finished_utc"].replace("Z", "+00:00"))
    duration = (finish - start).total_seconds()

report = {
    "run_id": metadata["run_id"], "started_utc": metadata["started_utc"],
    "finished_utc": metadata.get("finished_utc"), "duration_seconds": duration,
    "gpu": gpu, "container": container, "gpu_process_peak_memory_mib": dict(sorted(gpu_processes.items())),
    "collector_errors": {"count": len(error_lines), "categories": dict(sorted(categories.items()))},
    "container_final_state": {
        "running": metadata.get("container_running"), "restart_count": metadata.get("container_restart_count"),
        "oom_killed": metadata.get("container_oom_killed"),
    },
    "warnings": sorted(set(warnings)),
}
(run / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
lines = [
    "# MT Cloud Monitor Report", "", f"- Run: `{report['run_id']}`",
    f"- Duration: `{report['duration_seconds']}` seconds", f"- Collector errors: `{report['collector_errors']['count']}`",
    "", "## GPU", "", f"- Samples: `{gpu['sample_count']}`",
    f"- Utilization average/P95/max: `{gpu['utilization_percent']['average']}` / `{gpu['utilization_percent']['p95']}` / `{gpu['utilization_percent']['maximum']}`",
    f"- Memory used max MiB: `{gpu['memory_used_mib']['maximum']}`",
    "", "## Container", "", f"- Samples: `{container['sample_count']}`",
    f"- CPU average/P95/max: `{container['cpu_percent']['average']}` / `{container['cpu_percent']['p95']}` / `{container['cpu_percent']['maximum']}`",
    f"- Memory used max bytes: `{container['memory_used_bytes']['maximum']}`",
    "", "## Warnings", "",
]
lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- None"])
lines.extend(["", "Request throughput, latency, token rate, and cost remain in the separate local MT benchmark report.", ""])
(run / "report.md").write_text("\n".join(lines))
PY
}

write_manifest() {
  (
    cd "${CURRENT_DIR}"
    while IFS= read -r -d '' file; do sha256sum "${file#./}"; done < <(
      find . -maxdepth 1 -type f ! -name manifest.sha256 ! -name .completed -print0 | sort -z
    )
  ) >"${CURRENT_DIR}/manifest.sha256"
}

apply_retention() {
  local path archive
  while IFS= read -r -d '' path; do
    [[ -f "${path}/.completed" ]] || continue
    archive="${RUNS_DIR}/$(basename "${path}").tar.gz"
    rm -rf -- "${path}"
    [[ -f "${archive}" ]] && rm -- "${archive}"
  done < <(find "${RUNS_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime "+${KEEP_DAYS}" -print0)

  mapfile -t completed_runs < <(find "${RUNS_DIR}" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/.completed' ';' -print | sort)
  while [[ ${#completed_runs[@]} -gt ${KEEP_RUNS} ]]; do
    path="${completed_runs[0]}"
    archive="${RUNS_DIR}/$(basename "${path}").tar.gz"
    rm -rf -- "${path}"
    [[ -f "${archive}" ]] && rm -- "${archive}"
    completed_runs=("${completed_runs[@]:1}")
  done
}

finalize() {
  local requested_status="${1:-0}" final_status finished_utc
  [[ ${FINALIZED} -eq 0 ]] || return 0
  FINALIZED=1
  trap - INT TERM EXIT
  stop_collectors
  finished_utc="$(utc_now)"
  final_status="${requested_status}"
  write_metadata "${finished_utc}" || { record_error finalize metadata_failed; final_status=1; }
  generate_reports || { record_error finalize report_failed; final_status=1; }
  if [[ ${final_status} -eq 0 ]]; then
    write_manifest || { record_error finalize manifest_failed; final_status=1; }
  fi
  if [[ ${final_status} -eq 0 ]]; then
    : >"${CURRENT_DIR}/.completed"
    if tar -C "${RUNS_DIR}" -czf "${PARTIAL_ARCHIVE}" "${RUN_ID}"; then
      mv -- "${PARTIAL_ARCHIVE}" "${ARCHIVE_PATH}"
      apply_retention
    else
      record_error finalize archive_failed
      rm -f -- "${CURRENT_DIR}/.completed" "${PARTIAL_ARCHIVE}"
      final_status=1
    fi
  fi
  release_lock
  echo
  echo "MT benchmark monitor stopped."
  echo "Run directory: ${CURRENT_DIR}"
  echo "Report: ${CURRENT_DIR}/report.md"
  [[ -f "${ARCHIVE_PATH}" ]] && echo "Archive to provide: ${ARCHIVE_PATH}"
  exit "${final_status}"
}

for command in python3 docker nvidia-smi tar sha256sum find sort sed grep hostname uname git; do require_command "${command}"; done
validate_nonnegative_integer MT_MONITOR_GPU_INDEX "${GPU_INDEX}"
validate_positive_integer MT_MONITOR_KEEP_RUNS "${KEEP_RUNS}"
validate_positive_integer MT_MONITOR_KEEP_DAYS "${KEEP_DAYS}"
validate_positive_decimal MT_MONITOR_GPU_INTERVAL_SECONDS "${GPU_INTERVAL}"
validate_positive_decimal MT_MONITOR_CONTAINER_INTERVAL_SECONDS "${CONTAINER_INTERVAL}"
prepare_output
resolve_container
if ! nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "Unable to query selected GPU index: ${GPU_INDEX}" >&2
  release_lock
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
CURRENT_DIR="${RUNS_DIR}/${RUN_ID}"
ARCHIVE_PATH="${RUNS_DIR}/${RUN_ID}.tar.gz"
PARTIAL_ARCHIVE="${ARCHIVE_PATH}.partial"
STARTED_UTC="$(utc_now)"
mkdir -- "${CURRENT_DIR}"
: >"${CURRENT_DIR}/collector-errors.log"
: >"${CURRENT_DIR}/service.log"
write_metadata "" || { release_lock; exit 2; }

trap 'finalize 0' INT TERM
trap 'finalize $?' EXIT
start_collector gpu_collector
start_collector gpu_process_collector
start_collector container_collector
start_collector service_log_collector

echo "MT benchmark monitor started."
echo "Run ID: ${RUN_ID}"
echo "Run directory: ${CURRENT_DIR}"
echo "Run the MT benchmark from another machine, then press Ctrl+C here."
while true; do sleep 1; done
