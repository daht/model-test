import json
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
printf 'docker %s\n' "$*" >>"${FAKE_CALL_LOG}"
if [[ "$1" == "compose" && "$2" == "ps" ]]; then
  service="${4:-}"
  if [[ "${service}" == "hy-mt-api" ]]; then
    ids="${FAKE_GATEWAY_IDS:-gateway-container}"
  else
    ids="${FAKE_VLLM_IDS:-vllm-container}"
  fi
  if [[ "${ids}" != "__EMPTY__" ]]; then
    printf '%b\n' "${ids}"
  fi
elif [[ "$1" == "inspect" ]]; then
  format="$3"
  case "$format" in
    *Image*) echo image-id ;;
    *StartedAt*) echo 2026-07-20T00:00:00Z ;;
    *Running*) echo true ;;
    *RestartCount*) echo 0 ;;
    *OOMKilled*) echo false ;;
    *) echo unknown ;;
  esac
elif [[ "$1" == "stats" ]]; then
  if [[ "${@: -1}" == "gateway-container" ]]; then
    echo '12.5%|512MiB / 2GiB|25.0%|1kB / 2kB|3kB / 4kB|5'
  else
    echo '75.0%|4GiB / 8GiB|50.0%|2kB / 4kB|6kB / 8kB|10'
  fi
elif [[ "$1" == "compose" && "$2" == "logs" ]]; then
  echo '2026-07-20T00:00:01Z 198.51.100.42:54321 service log'
  while true; do sleep 1; done
else
  exit 1
fi
""",
    )
    executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
set -eu
printf 'nvidia-smi %s\n' "$*" >>"${FAKE_CALL_LOG}"
if [[ "$*" == *query-compute-apps* ]]; then
  echo '123, python, 2048'
elif [[ "$*" == *query-gpu=index,name* ]]; then
  if [[ "$*" == *utilization.gpu* ]]; then
    [[ "${FAKE_NO_GPU_SAMPLES:-0}" == "1" ]] || echo '0, NVIDIA A10, 60, 30, 12000, 23028, 110, 65, P0, 1500, 6000'
  else
    echo '0, NVIDIA A10'
  fi
else
  exit 1
fi
""",
    )
    return fake_bin, call_log


def start_monitor(tmp_path: Path, **overrides) -> tuple[subprocess.Popen, Path, Path]:
    fake_bin, call_log = fake_tools(tmp_path)
    output_root = tmp_path / "monitor"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_CALL_LOG": str(call_log),
        "MT_MONITOR_OUTPUT_ROOT": str(output_root),
        "MT_MONITOR_GPU_INTERVAL_SECONDS": "0.05",
        "MT_MONITOR_CONTAINER_INTERVAL_SECONDS": "0.05",
        **overrides,
    }
    process = subprocess.Popen(
        ["bash", "scripts/monitor_mt_benchmark.sh"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, output_root, call_log


def wait_for_started(process: subprocess.Popen) -> str:
    deadline = time.monotonic() + 5
    output = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        output.append(line)
        if "MT benchmark monitor started." in line:
            return "".join(output)
        if process.poll() is not None:
            break
    process.kill()
    raise AssertionError("monitor did not start: " + "".join(output))


def wait_for_data_row(output_root: Path, filename: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        runs = [path for path in (output_root / "runs").iterdir() if path.is_dir()]
        if len(runs) == 1:
            evidence = runs[0] / filename
            if evidence.is_file() and len(evidence.read_text().splitlines()) >= 2:
                return
        time.sleep(0.01)
    raise AssertionError(f"monitor did not collect a data row for {filename}")


def test_monitor_lifecycle_collects_reports_and_archive(tmp_path):
    process, output_root, call_log = start_monitor(tmp_path)
    started_output = wait_for_started(process)
    wait_for_data_row(output_root, "gpu.csv")
    wait_for_data_row(output_root, "gateway-container.csv")
    wait_for_data_row(output_root, "vllm-container.csv")
    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=10)

    assert process.returncode == 0, started_output + stdout
    runs = [path for path in (output_root / "runs").iterdir() if path.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    for name in (
        "metadata.json",
        "gpu.csv",
        "gpu-processes.csv",
        "gateway-container.csv",
        "vllm-container.csv",
        "gateway-service.log",
        "vllm-service.log",
        "collector-errors.log",
        "report.json",
        "report.md",
        "manifest.sha256",
        ".completed",
    ):
        assert (run / name).is_file(), name
    assert (output_root / "runs" / f"{run.name}.tar.gz").is_file()
    assert not (output_root / ".monitor.lock").exists()

    report = json.loads((run / "report.json").read_text())
    assert report["gpu"]["sample_count"] >= 1
    assert report["gpu"]["utilization_percent"]["maximum"] == 60.0
    assert report["gpu"]["memory_used_mib"]["maximum"] == 12000.0
    assert report["containers"]["gateway"]["cpu_percent"]["maximum"] == 12.5
    assert report["containers"]["gateway"]["memory_used_bytes"]["maximum"] == 512 * 1024**2
    assert report["containers"]["vllm"]["cpu_percent"]["maximum"] == 75.0
    assert report["containers"]["vllm"]["memory_used_bytes"]["maximum"] == 4 * 1024**3
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["services"]["gateway"]["service"] == "hy-mt-api"
    assert metadata["services"]["vllm"]["service"] == "hy-mt-vllm"

    calls = call_log.read_text()
    assert "curl" not in calls
    assert "wget" not in calls
    assert "restart" not in calls
    assert " compose up" not in calls
    assert " compose down" not in calls
    evidence = "".join(path.read_text(errors="replace") for path in run.iterdir() if path.is_file())
    assert "API_KEY" not in evidence
    assert "MT_BENCHMARK_URL" not in evidence
    assert "198.51.100." not in evidence
    assert "[redacted-ip]" in evidence
    manifest = (run / "manifest.sha256").read_text()
    assert "report.json" in manifest
    assert "metadata.json" in manifest
    assert "manifest.sha256" not in manifest


def test_monitor_rejects_missing_container(tmp_path):
    process, _, _ = start_monitor(tmp_path, FAKE_VLLM_IDS="__EMPTY__")
    stdout, _ = process.communicate(timeout=5)

    assert process.returncode == 2
    assert "exactly one running MT container" in stdout


def test_monitor_rejects_multiple_containers(tmp_path):
    process, _, _ = start_monitor(tmp_path, FAKE_GATEWAY_IDS="one\\ntwo")
    stdout, _ = process.communicate(timeout=5)

    assert process.returncode == 2
    assert "exactly one running MT container" in stdout


def test_monitor_refuses_unmarked_nonempty_output_root(tmp_path):
    output_root = tmp_path / "monitor"
    output_root.mkdir()
    (output_root / "foreign.txt").write_text("keep")
    process, _, _ = start_monitor(tmp_path)
    stdout, _ = process.communicate(timeout=5)

    assert process.returncode == 2
    assert "unmarked" in stdout
    assert (output_root / "foreign.txt").read_text() == "keep"


def test_monitor_reports_missing_gpu_samples_as_null(tmp_path):
    process, output_root, _ = start_monitor(tmp_path, FAKE_NO_GPU_SAMPLES="1")
    wait_for_started(process)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=10)

    run = next(path for path in (output_root / "runs").iterdir() if path.is_dir())
    report = json.loads((run / "report.json").read_text())
    assert report["gpu"]["sample_count"] == 0
    assert report["gpu"]["utilization_percent"]["average"] is None
    assert "missing GPU samples" in report["warnings"]


def test_monitor_documentation_is_safe_and_complete():
    documentation = (ROOT / "docs" / "mt-cloud-monitor.md").read_text()
    for value in (
        "scripts/monitor_mt_benchmark.sh",
        "hy-mt-api",
        "hy-mt-vllm",
        "/tmp/mt-monitor",
        "MT_MONITOR_GATEWAY_SERVICE",
        "MT_MONITOR_VLLM_SERVICE",
        "MT_MONITOR_GPU_INDEX",
        "MT_MONITOR_OUTPUT_ROOT",
        "MT_MONITOR_GPU_INTERVAL_SECONDS",
        "MT_MONITOR_CONTAINER_INTERVAL_SECONDS",
        "MT_MONITOR_KEEP_RUNS",
        "MT_MONITOR_KEEP_DAYS",
        "Ctrl+C",
        "report.json",
        "report.md",
        "tar.gz",
        "不会发送翻译请求",
        "服务日志",
        "gateway-container.csv",
        "vllm-container.csv",
    ):
        assert value in documentation
    assert "API_KEY=" not in documentation
    assert "MT_BENCHMARK_URL=" not in documentation
    assert "http://" not in documentation
    assert "https://" not in documentation
    assert "198.51.100." not in documentation
