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
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -eu
echo 'vllm:omni_num_requests_waiting{model_name="Qwen"} 0'
""",
    )
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
elif [[ "$1" == "logs" ]]; then
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
    upstream_log = tmp_path / "vllm-omni.log"
    upstream_log.write_text(
        "[SpeechE2E] request_id=speech-1 stream=true status=ok total_ms=20.0 first_chunk_ms=5.0\\n"
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TTS_MONITOR_OUTPUT_ROOT": str(output_root),
        "TTS_MONITOR_GPU_INTERVAL_SECONDS": "0.05",
        "TTS_MONITOR_CONTAINER_INTERVAL_SECONDS": "0.05",
        "TTS_MONITOR_VLLM_OMNI_METRICS_URL": "http://127.0.0.1:8091/metrics",
        "TTS_MONITOR_VLLM_OMNI_METRICS_INTERVAL_SECONDS": "0.05",
        "TTS_MONITOR_VLLM_OMNI_LOG": str(upstream_log),
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
        "vllm-omni.log",
        "vllm-omni-metrics.prom",
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
    assert (run / "service.log").stat().st_size > 0
    assert "198.51.100." not in (run / "service.log").read_text()
    assert "[redacted-ip]" in (run / "service.log").read_text()
    assert "[SpeechE2E]" in (run / "vllm-omni.log").read_text()
    assert "vllm:omni_num_requests_waiting" in (run / "vllm-omni-metrics.prom").read_text()


def test_triton_metrics_analyzer_reports_per_model_batching(tmp_path):
    metrics = tmp_path / "triton-metrics.prom"
    metrics.write_text(
        """# sampled_at=2026-08-01T00:00:00.000Z
nv_inference_request_success{model=\"llm\",version=\"1\"} 10
nv_inference_exec_count{model=\"llm\",version=\"1\"} 10
nv_inference_count{model=\"llm\",version=\"1\"} 10
nv_inference_request_success{model=\"vocoder\",version=\"1\"} 8
nv_inference_exec_count{model=\"vocoder\",version=\"1\"} 8
nv_inference_count{model=\"vocoder\",version=\"1\"} 8
# sampled_at=2026-08-01T00:00:01.000Z
nv_inference_request_success{model=\"llm\",version=\"1\"} 16
nv_inference_exec_count{model=\"llm\",version=\"1\"} 12
nv_inference_count{model=\"llm\",version=\"1\"} 18
nv_inference_request_success{model=\"vocoder\",version=\"1\"} 14
nv_inference_exec_count{model=\"vocoder\",version=\"1\"} 14
nv_inference_count{model=\"vocoder\",version=\"1\"} 14
"""
    )
    result = subprocess.run(
        [
            "python3",
            "scripts/analyze_triton_metrics.py",
            str(metrics),
            "--json-output",
            str(tmp_path / "report.json"),
            "--markdown-output",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        check=True,
    )
    assert result.returncode == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["snapshot_count"] == 2
    assert report["models"] == [
        {
            "average_batch_size": 4.0,
            "executions_delta": 2.0,
            "inference_elements_delta": 8.0,
            "model": "llm",
            "requests_per_execution": 3.0,
            "successful_requests_delta": 6.0,
        },
        {
            "average_batch_size": 1.0,
            "executions_delta": 6.0,
            "inference_elements_delta": 6.0,
            "model": "vocoder",
            "requests_per_execution": 1.0,
            "successful_requests_delta": 6.0,
        },
    ]
    assert "| vocoder | 6.00 | 6.00 | 6.00 | 1.00 | 1.00 |" in (tmp_path / "report.md").read_text()


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
        "TTS_MONITOR_VLLM_OMNI_METRICS_URL",
        "TTS_MONITOR_VLLM_OMNI_LOG",
        "TTS_MONITOR_TRITON_SERVICE",
        "TTS_MONITOR_TRITON_METRICS_URL",
        "TTS_MONITOR_VOXSERVE_SERVICE",
        "TTS_MONITOR_VOXSERVE_LOG",
        "Ctrl+C",
        "does not require API_KEY",
    ):
        assert value in result.stdout
