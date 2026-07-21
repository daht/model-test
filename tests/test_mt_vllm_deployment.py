from pathlib import Path
import os
import subprocess

import yaml


ROOT = Path(__file__).parents[1]
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.23.0-cu129@"
    "sha256:dc426b7b77cbc1bd833f97a807c6adac0e2d241d19216149e049f0e776795e5d"
)


def test_compose_defines_private_pinned_vllm_service():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["hy-mt-vllm"]

    assert service["image"] == VLLM_IMAGE
    assert service["profiles"] == ["mt-vllm"]
    assert "ports" not in service
    assert service["gpus"] == "all"
    assert "./models:/models:ro" in service["volumes"]
    assert service["ipc"] == "host"


def test_compose_uses_a10_vllm_limits_and_healthcheck():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["hy-mt-vllm"]
    command = service["command"]

    assert command == [
        "--model",
        "/models/Hy-MT2-1.8B",
        "--served-model-name",
        "/models/Hy-MT2-1.8B",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "32",
    ]
    healthcheck = " ".join(service["healthcheck"]["test"])
    assert "127.0.0.1:8000/health" in healthcheck


def test_gateway_dependency_is_optional_outside_vllm_profile():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    dependency = compose["services"]["hy-mt-api"]["depends_on"]["hy-mt-vllm"]

    assert dependency == {"condition": "service_healthy", "required": False}


def test_mt_vllm_environment_example_is_safe_and_focused():
    content = (ROOT / "cloud" / "A10.mt-vllm.env.example").read_text()

    for expected in (
        "API_KEY=",
        "MODEL_NAME=Hy-MT2-1.8B",
        "MODEL_BACKEND=vllm",
        "VLLM_BASE_URL=http://hy-mt-vllm:8000",
        "VLLM_MODEL=/models/Hy-MT2-1.8B",
        "VLLM_TIMEOUT_SECONDS=120",
        "MAX_NEW_TOKENS=1024",
    ):
        assert expected in content
    api_key_line = next(line for line in content.splitlines() if line.startswith("API_KEY="))
    assert api_key_line == "API_KEY="
    assert "118.195." not in content
    assert "ASR_" not in content
    assert "TTS_" not in content


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_deploy_script(tmp_path: Path, include_env: bool = True):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    if include_env:
        (tmp_path / ".env").write_text("API_KEY=test-only\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
printf 'docker %s\n' "$*" >>"${CALL_LOG}"
if [[ "$1 $2" == "compose version" ]]; then exit 0; fi
if [[ "$1 $2" == "compose exec" ]]; then exit 0; fi
if [[ "$1 $2" == "compose logs" ]]; then exit 0; fi
if [[ "$1" == "compose" ]]; then exit 0; fi
exit 1
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -eu
printf 'curl %s\n' "$*" >>"${CALL_LOG}"
printf '{"status":"ok"}\n'
""",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "HEALTH_TIMEOUT_SECONDS": "1",
        "HEALTH_INTERVAL_SECONDS": "0.01",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "deploy_mt_vllm.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, call_log


def test_deploy_script_starts_vllm_before_rebuilding_gateway(tmp_path):
    result, call_log = _run_deploy_script(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text().splitlines()
    expected_in_order = [
        "docker compose --profile mt-vllm pull hy-mt-vllm",
        "docker compose --profile mt-vllm up -d hy-mt-vllm",
        "docker compose --profile mt-vllm exec -T hy-mt-vllm",
        "docker compose --profile mt-vllm build hy-mt-api",
        "docker compose --profile mt-vllm up -d --force-recreate --no-deps hy-mt-api",
        "curl -fsS http://127.0.0.1:8000/health",
    ]
    positions = []
    for expected in expected_in_order:
        positions.append(next(index for index, call in enumerate(calls) if expected in call))
    assert positions == sorted(positions)
    evidence = "\n".join(calls)
    for forbidden in (
        "qwen-asr-api",
        "cosyvoice-tts-api",
        " stop ",
        " down ",
        " restart ",
    ):
        assert forbidden not in f" {evidence} "


def test_deploy_script_requires_environment_file(tmp_path):
    result, call_log = _run_deploy_script(tmp_path, include_env=False)

    assert result.returncode != 0
    assert "Missing .env" in result.stdout
    assert not call_log.exists()
