import json
import os
import subprocess
import sys
from pathlib import Path


HELPER = Path("scripts/asr_deploy_receipt.py").resolve()
API_KEY = "unit-test-production-key-0123456789abcdef"
COMMAND = ["python", "-m", "uvicorn", "app.asr_api:app", "--workers", "1"]


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, str(HELPER), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _baseline_files(
    tmp_path,
    *,
    container_backend="qwen_vllm",
    started_at="2099-01-01T00:00:00Z",
):
    repo = tmp_path / "repo"
    model_root = repo / "models"
    model = model_root / "Qwen3-ASR-1.7B"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    manifest = model_root / "Qwen3-ASR-1.7B.manifest.json"
    manifest.write_text('{"approved":true}\n')
    env_file = repo / ".env"
    env_file.write_text(f"API_KEY={API_KEY}\n")
    environment = {
        "API_KEY": API_KEY,
        "ASR_BACKEND": "qwen_vllm",
        "ASR_STREAM_MODE": "stateful",
        "ASR_REQUIRE_MODEL_MANIFEST": "true",
        "ASR_EAGER_LOAD": "true",
        "ASR_FILE_TRANSCRIBE_ENABLED": "false",
        "ASR_MODEL_ID": "/models/Qwen3-ASR-1.7B",
        "ASR_MODEL_MANIFEST_PATH": "/models/Qwen3-ASR-1.7B.manifest.json",
    }
    compose = tmp_path / "compose.json"
    compose.write_text(
        json.dumps(
            {
                "services": {
                    "qwen-asr-api": {
                        "environment": environment,
                        "command": COMMAND,
                    }
                }
            }
        )
    )
    container_environment = dict(environment)
    container_environment["ASR_BACKEND"] = container_backend
    inspect = tmp_path / "inspect.json"
    inspect.write_text(
        json.dumps(
            [
                {
                    "Image": "sha256:" + "1" * 64,
                    "Config": {
                        "Env": [
                            f"{name}={value}"
                            for name, value in container_environment.items()
                        ],
                        "Cmd": COMMAND,
                        "Labels": {"com.docker.compose.config-hash": "config-hash"},
                    },
                    "State": {"Running": True, "StartedAt": started_at},
                    "Mounts": [
                        {
                            "Destination": "/models",
                            "Source": str(model_root),
                            "RW": False,
                        }
                    ],
                }
            ]
        )
    )
    return repo, model, manifest, env_file, compose, inspect


def _validate_baseline(repo, model, manifest, env_file, compose, inspect):
    return _run(
        "validate-baseline",
        "--compose-config",
        compose,
        "--container-inspect",
        inspect,
        "--repository",
        repo,
        "--compose-hash",
        "config-hash",
        "--env-file",
        env_file,
        "--model-dir",
        model,
        "--manifest",
        manifest,
    )


def test_running_baseline_accepts_matching_config_model_mount_and_start_state(tmp_path):
    inputs = _baseline_files(tmp_path)
    result = _validate_baseline(*inputs)

    assert result.returncode == 0, result.stderr
    assert API_KEY not in result.stdout
    assert API_KEY not in result.stderr


def test_running_baseline_reads_compose_hash_and_inspection_without_secret_output(tmp_path):
    repo, model, manifest, env_file, compose, inspect = _baseline_files(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "$*" == *'config --format json'* ]]; then cat "$COMPOSE_JSON"; exit 0; fi
if [[ "$*" == *'config --hash qwen-asr-api'* ]]; then echo 'qwen-asr-api config-hash'; exit 0; fi
if [[ "${1:-}" == inspect ]]; then cat "$INSPECT_JSON"; exit 0; fi
exit 1
"""
    )
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "COMPOSE_JSON": str(compose),
        "INSPECT_JSON": str(inspect),
    }

    result = _run(
        "validate-running-baseline",
        "--container-id",
        "old-container",
        "--repository",
        repo,
        "--env-file",
        env_file,
        "--model-dir",
        model,
        "--manifest",
        manifest,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert API_KEY not in result.stdout
    assert API_KEY not in result.stderr


def test_running_baseline_rejects_prechanged_candidate_environment_without_secret_output(tmp_path):
    inputs = _baseline_files(tmp_path, container_backend="mock")
    result = _validate_baseline(*inputs)

    assert result.returncode != 0
    assert "ASR_BACKEND" in result.stderr
    assert API_KEY not in result.stdout
    assert API_KEY not in result.stderr


def test_running_baseline_rejects_unmatched_full_compose_service_config(tmp_path):
    inputs = _baseline_files(tmp_path)
    inspect_path = inputs[-1]
    payload = json.loads(inspect_path.read_text())
    payload[0]["Config"]["Labels"]["com.docker.compose.config-hash"] = "old-config"
    inspect_path.write_text(json.dumps(payload))

    result = _validate_baseline(*inputs)

    assert result.returncode != 0
    assert "full Compose service hash" in result.stderr


def test_running_baseline_rejects_model_or_manifest_changed_after_container_start(tmp_path):
    inputs = _baseline_files(tmp_path, started_at="2000-01-01T00:00:00Z")
    result = _validate_baseline(*inputs)

    assert result.returncode != 0
    assert "changed after the container started" in result.stderr


def test_receipt_binds_image_config_manifest_model_and_evidence(tmp_path):
    repo, model, manifest, env_file, _, _ = _baseline_files(tmp_path)
    protected = tmp_path / "secure"
    protected.mkdir()
    evidence = protected / "release.log"
    evidence.write_text("ASR release verification passed.\n")
    receipt = protected / "release.receipt.json"
    candidate_sha = "a" * 40
    image_id = "sha256:" + "b" * 64

    created = _run(
        "create-receipt",
        "--kind",
        "release",
        "--output",
        receipt,
        "--repository",
        repo,
        "--candidate-sha",
        candidate_sha,
        "--image-id",
        image_id,
        "--env-file",
        env_file,
        "--model-dir",
        model,
        "--manifest",
        manifest,
        "--evidence",
        evidence,
    )
    assert created.returncode == 0, created.stderr
    assert receipt.stat().st_mode & 0o777 == 0o600

    validate_args = (
        "validate-receipt",
        "--kind",
        "release",
        "--receipt",
        receipt,
        "--repository",
        repo,
        "--candidate-sha",
        candidate_sha,
        "--image-id",
        image_id,
        "--env-file",
        env_file,
        "--model-dir",
        model,
        "--manifest",
        manifest,
    )
    assert _run(*validate_args).returncode == 0

    image_mismatch_args = list(validate_args)
    image_index = image_mismatch_args.index("--image-id") + 1
    image_mismatch_args[image_index] = "sha256:" + "c" * 64
    image_rejected = _run(*image_mismatch_args)
    assert image_rejected.returncode != 0
    assert "image_id" in image_rejected.stderr

    env_file.write_text(f"API_KEY={API_KEY}\nASR_STREAM_MODE=changed\n")
    rejected = _run(*validate_args)
    assert rejected.returncode != 0
    assert "env_sha256" in rejected.stderr
    assert API_KEY not in rejected.stdout
    assert API_KEY not in rejected.stderr
