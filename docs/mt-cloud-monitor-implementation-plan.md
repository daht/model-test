# MT Cloud Benchmark Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one cloud-executable Shell monitor that observes an existing HY-MT Docker Compose service during an externally generated benchmark and produces a complete summary archive without sending MT requests.

**Architecture:** `scripts/monitor_mt_benchmark.sh` owns validation, lock/ownership, four background collectors, idempotent trap finalization, embedded-Python reporting, checksums, archive creation, and retention. Pytest replaces Docker, NVIDIA tools, and tar with deterministic fakes and signals the real Shell process. The ASR delivery workflow overrides per-task commit suggestions: one Development Agent owns all tasks, stages only intended files, runs the staged commit gate, and creates one final candidate commit.

**Tech Stack:** Bash, Docker Compose v2 CLI, `nvidia-smi`, Python 3 standard library, tar, sha256sum, pytest.

---

## Execution Contract

- Objective: deliver the approved single-file, monitor-only cloud workflow.
- Non-goals: no service calls, restarts, builds, config edits, Compose edits, ASR/TTS changes, or request-content logging.
- Risk: monitor overhead, leaked child processes, incomplete archives, unsafe retention, and secret/topology leakage.
- Intended paths: `scripts/monitor_mt_benchmark.sh`, `tests/test_monitor_mt_benchmark.py`, `docs/mt-cloud-monitor.md`, `README.md`.
- Focused test: `/model/.venv/bin/python -m pytest -q tests/test_monitor_mt_benchmark.py`.
- Candidate gate: explicitly stage intended paths, then `scripts/verify_asr_release.sh commit`, then commit.
- External gate: real Docker, A10, HY-MT container, benchmark window, and returned archive remain unavailable locally.
- Checkpoints: report evidence and remaining hypotheses at 30 minutes; reduce scope if no testable staged candidate at 60 minutes.

## File Responsibilities

- `scripts/monitor_mt_benchmark.sh`: the only cloud runtime artifact.
- `tests/test_monitor_mt_benchmark.py`: fake-tool process tests and report assertions.
- `docs/mt-cloud-monitor.md`: operator start/stop/retrieval instructions and evidence limits.
- `README.md`: one file-inventory entry.

### Task 1: Ownership, validation, and signal-safe lifecycle

**Files:**
- Create: `tests/test_monitor_mt_benchmark.py`
- Create: `scripts/monitor_mt_benchmark.sh`

- [ ] **Step 1: Write the failing lifecycle test**

Create fake executable helpers under `tmp_path/bin`. The fake `docker` returns one container for `docker compose ps -q hy-mt-api`, deterministic inspect fields, one stats row, and a follow loop for logs. Fake `nvidia-smi` returns one GPU row and one process row. Fake `tar` delegates to the system tar path captured before PATH replacement.

Launch the monitor with:

```python
process = subprocess.Popen(
    ["bash", "scripts/monitor_mt_benchmark.sh"],
    cwd=REPOSITORY_ROOT,
    env={
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "MT_MONITOR_OUTPUT_ROOT": str(tmp_path / "monitor"),
        "MT_MONITOR_GPU_INTERVAL_SECONDS": "0.05",
        "MT_MONITOR_CONTAINER_INTERVAL_SECONDS": "0.05",
    },
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
```

Wait for `MT benchmark monitor started.`, send `SIGINT`, and assert exit zero, exactly one run directory, `.completed`, `report.json`, `report.md`, `manifest.sha256`, matching `.tar.gz`, and no lock directory.

- [ ] **Step 2: Run RED**

Run `/model/.venv/bin/python -m pytest -q tests/test_monitor_mt_benchmark.py -k lifecycle`.

Expected: fail because `scripts/monitor_mt_benchmark.sh` does not exist.

- [ ] **Step 3: Implement minimal safe lifecycle**

Use strict unset/pipe handling without global `set -e`, since collectors must record errors and continue:

```bash
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
PIDS=()
FINALIZED=0
```

Add exact functions: `require_command`, integer/decimal validators, `prepare_output`, `resolve_container`, `start_collector`, `stop_collectors`, and idempotent `finalize`. Enforce absolute/non-root/non-symlink output, marker ownership, atomic lock, exactly one running container, and selected-GPU query before run creation. Install traps only after validation; print the start marker only after collectors launch.

- [ ] **Step 4: Run GREEN**

Run the lifecycle test and require a clean pass with no remaining fake child processes.

### Task 2: Resource collectors and evidence boundaries

**Files:**
- Modify: `tests/test_monitor_mt_benchmark.py`
- Modify: `scripts/monitor_mt_benchmark.sh`

- [ ] **Step 1: Write failing collector tests**

Assert completed runs contain headers and rows for `gpu.csv`, `gpu-processes.csv`, and `container.csv`; assert `service.log` contains fake service lines. Record fake invocations and assert there is no `curl`, `wget`, HTTP URL, API key argument, restart, build, `up`, `down`, environment inspection, or command-line inspection.

Add exact refusal cases:

```python
assert_monitor_fails(env={"MT_MONITOR_OUTPUT_ROOT": "relative"}, message="absolute")
assert_monitor_fails(precreate_unmarked_nonempty=True, message="unmarked")
assert_monitor_fails(precreate_lock=True, message="Another MT monitor")
assert_monitor_fails(container_ids="", message="exactly one running MT container")
assert_monitor_fails(container_ids="one\ntwo", message="exactly one running MT container")
```

- [ ] **Step 2: Run RED**

Run the collector and validation subset; expect missing rows or missing refusal behavior.

- [ ] **Step 3: Implement collectors**

Implement the GPU collector exactly in this pattern:

```bash
gpu_collector() {
  local output="${CURRENT_DIR}/gpu.csv"
  echo 'sampled_at,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,power_watts,temperature_c,pstate,sm_clock_mhz,memory_clock_mhz' >"${output}"
  while true; do
    local sampled_at value
    sampled_at="$(utc_now)"
    if value="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,pstate,clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s,%s\n' "${sampled_at}" "${value}" >>"${output}"
    else
      record_error gpu sample_failed
    fi
    sleep "${GPU_INTERVAL}"
  done
}
```

Use the same error-continuation pattern for GPU processes and `docker stats --no-stream`. Follow `docker compose logs --follow --timestamps --since "${STARTED_UTC}" "${SERVICE}"`. Metadata uses allowlisted fields only and never inspects `.Config.Env` or process command lines.

- [ ] **Step 4: Run GREEN**

Run all Task 1-2 tests, then `bash -n scripts/monitor_mt_benchmark.sh`.

### Task 3: Reports, checksums, archive, and retention

**Files:**
- Modify: `tests/test_monitor_mt_benchmark.py`
- Modify: `scripts/monitor_mt_benchmark.sh`

- [ ] **Step 1: Write failing report and retention tests**

Feed fixed fake samples and assert sample counts, GPU average/P95/max, memory peak, power average/max, temperature max, container CPU average/P95/max, memory average/max, GPU-process peak memory, and collector error count.

Add missing-sample assertions:

```python
assert report["gpu"]["sample_count"] == 0
assert report["gpu"]["utilization_percent"]["average"] is None
assert "missing GPU samples" in report["warnings"]
```

Verify the manifest covers every regular evidence file except itself, `.completed` is absent on forced report/archive failure, `.partial` never remains after success, and retention never removes unowned, incomplete, or outside-root files.

- [ ] **Step 2: Run RED**

Run the report/retention subset; expect missing metrics and artifact-order failures.

- [ ] **Step 3: Implement embedded report generation and atomic finalization**

The embedded Python uses only the standard library and deterministic nearest rank:

```python
def nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]
```

Parse Docker byte units `B`, `kB`, `KiB`, `MB`, `MiB`, `GB`, and `GiB`; unsupported units create warnings and null aggregates. Write JSON, Markdown, manifest, `.completed`, `.partial` archive, atomic archive rename, then owned-artifact retention in that order.

- [ ] **Step 4: Run GREEN**

Run the complete focused test file and `bash -n`.

### Task 4: Documentation and staged candidate gate

**Files:**
- Create: `docs/mt-cloud-monitor.md`
- Modify: `README.md`
- Modify: `tests/test_monitor_mt_benchmark.py`

- [ ] **Step 1: Write failing documentation contract test**

Require script path, defaults, start/Ctrl+C flow, report/archive paths, override names, no-request statement, and service-log content warning. Reject endpoint URLs, API-key assignments, public IP literals, and restart/build commands.

- [ ] **Step 2: Run RED**

Run the documentation test; expect failure because `docs/mt-cloud-monitor.md` is absent.

- [ ] **Step 3: Write the runbook and README link**

Document one safe command:

```bash
scripts/monitor_mt_benchmark.sh
```

Document overrides without infrastructure values and add one README inventory entry. Do not include an endpoint, key, corpus, or translation example.

- [ ] **Step 4: Run focused verification**

Run:

```bash
/model/.venv/bin/python -m pytest -q tests/test_monitor_mt_benchmark.py
bash -n scripts/monitor_mt_benchmark.sh
git diff --check
```

- [ ] **Step 5: Stage exact files and run candidate gate**

Run:

```bash
git add -- scripts/monitor_mt_benchmark.sh tests/test_monitor_mt_benchmark.py docs/mt-cloud-monitor.md README.md
scripts/verify_asr_release.sh commit
```

The gate must inspect the staged candidate. Do not stage already committed design/plan files again.

- [ ] **Step 6: Commit the candidate**

```bash
git commit -m "feat(mt): add cloud benchmark monitor"
git status --short
```

Expected: clean candidate worktree.

## Independent and Product Acceptance

- [ ] Dispatch one separate read-only Test Agent against the exact candidate SHA. It runs focused and adversarial fake-tool probes, `bash -n`, and the commit runner, then returns `ACCEPTED` or `REJECTED`.
- [ ] Return rejection findings to the same Development Agent and repeat independent acceptance on the corrected SHA.
- [ ] After acceptance, the primary agent independently runs the focused suite, operator lifecycle probe, syntax check, commit gate, secret/artifact scan, and final Git status.
- [ ] Report local evidence and candidate SHA. Real cloud Docker/A10/archive evidence remains unexecuted until the operator runs the script.
