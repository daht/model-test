import os
import signal
import subprocess
import time
from pathlib import Path


def executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_monitor_interrupt_finalizes_report_manifest_and_archive(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
if [[ "$1" == "compose" && "$2" == "ps" ]]; then
  [[ "${@: -1}" == "qwen-asr-api" ]] && echo asr-container || echo hymt-container
elif [[ "$1" == "inspect" ]]; then
  format="$3"; container="$4"
  case "$format" in
    *Image*) echo image-id ;;
    *Config.Cmd*) echo '["python"]' ;;
    *State.Pid*) [[ "$container" == "asr-container" ]] && echo 111 || echo 222 ;;
  esac
elif [[ "$1" == "stats" ]]; then
  echo 'qwen-asr-api-1|1.0%|100MiB / 1GiB|10%|0B / 0B|0B / 0B|2'
  echo 'hy-mt-api-1|2.0%|200MiB / 1GiB|20%|0B / 0B|0B / 0B|3'
elif [[ "$1" == "compose" && "$2" == "logs" ]]; then
  while true; do sleep 1; done
else
  exit 1
fi
""",
    )
    executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$*" == *query-compute-apps* ]]; then
  echo '111, python, 1024'
else
  echo '0, NVIDIA A10, 50, 20, 12000, 23028, 100, 60, P0, 1200, 6000'
fi
""",
    )
    output = tmp_path / "monitor"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "API_KEY": "unit-test-only-monitor-key-000000000000",
        "ASR_MONITOR_OUTPUT_ROOT": str(output),
        "ASR_MONITOR_HTTP_INTERVAL_SECONDS": "0.1",
        "ASR_MONITOR_GPU_INTERVAL_SECONDS": "0.1",
        "ASR_MONITOR_CONTAINER_INTERVAL_SECONDS": "0.1",
    }
    process = subprocess.Popen(
        ["bash", "scripts/monitor_asr_bottleneck.sh"],
        cwd=Path(__file__).parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5
    output_lines = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        output_lines.append(line)
        if "ASR bottleneck monitor started." in line:
            break
    else:
        process.kill()
        raise AssertionError("monitor did not start: " + "".join(output_lines))

    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=10)

    assert process.returncode == 0, "".join(output_lines) + stdout
    runs = [path for path in (output / "runs").iterdir() if path.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    assert (run / "report.json").is_file()
    assert (run / "report.md").is_file()
    assert (run / "manifest.sha256").is_file()
    assert (run / ".completed").is_file()
    assert (output / "runs" / f"{run.name}.tar.gz").is_file()
    assert not (output / ".monitor.lock").exists()
