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


def test_tts_monitor_collects_and_archives_evidence(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
if [[ "$1" == "compose" && "$2" == "ps" ]]; then
  echo tts-container
elif [[ "$1" == "inspect" ]]; then
  case "$3" in
    *Image*) echo image-id ;;
    *StartedAt*) echo 2026-07-24T00:00:00Z ;;
    *Running*) echo true ;;
    *RestartCount*) echo 0 ;;
    *OOMKilled*) echo false ;;
  esac
elif [[ "$1" == "stats" ]]; then
  echo '75.0%|4GiB / 8GiB|50.0%|2kB / 4kB|6kB / 8kB|10'
elif [[ "$1" == "compose" && "$2" == "logs" ]]; then
  echo '2026-07-24T00:00:01Z 198.51.100.42:1234 request'
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
if [[ "$*" == *query-compute-apps* ]]; then
  echo '123, python, 2048'
elif [[ "$*" == *utilization.gpu* ]]; then
  echo '0, NVIDIA A10, 60, 30, 12000, 23028, 110, 65, P0, 1500, 6000'
else
  echo '0, NVIDIA A10'
fi
""",
    )
    output_root = tmp_path / "monitor"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TTS_MONITOR_OUTPUT_ROOT": str(output_root),
        "TTS_MONITOR_GPU_INTERVAL_SECONDS": "0.05",
        "TTS_MONITOR_CONTAINER_INTERVAL_SECONDS": "0.05",
    }
    process = subprocess.Popen(
        ["bash", "scripts/monitor_tts_benchmark.sh"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        output.append(line)
        if "TTS benchmark monitor started." in line:
            break
    else:
        process.kill()
        raise AssertionError("monitor did not start: " + "".join(output))

    time.sleep(0.2)
    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=10)
    assert process.returncode == 0, "".join(output) + stdout

    run = next(path for path in (output_root / "runs").iterdir() if path.is_dir())
    for name in (
        "metadata.json",
        "gpu.csv",
        "gpu-processes.csv",
        "container.csv",
        "service.log",
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
    assert report["gpu"]["utilization_percent"]["maximum"] == 60
    assert report["gpu"]["memory_used_mib"]["maximum"] == 12000
    assert report["container"]["cpu_percent"]["maximum"] == 75
    assert "198.51.100." not in (run / "service.log").read_text()
    assert "[redacted-ip]" in (run / "service.log").read_text()


def test_tts_monitor_help_documents_safe_usage():
    result = subprocess.run(
        ["bash", "scripts/monitor_tts_benchmark.sh", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for value in (
        "TTS_MONITOR_OUTPUT_ROOT",
        "TTS_MONITOR_SERVICE",
        "TTS_MONITOR_GPU_INDEX",
        "Ctrl+C",
        "does not require API_KEY",
    ):
        assert value in result.stdout
