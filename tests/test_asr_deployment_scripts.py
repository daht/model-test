from pathlib import Path


def script(name: str) -> str:
    return Path("scripts", name).read_text()


def test_update_all_services_builds_recreates_and_waits_for_asr_readiness():
    content = script("update_all_services.sh")

    assert "compose build hy-mt-api qwen-asr-api" in content
    assert 'compose up -d --force-recreate --no-deps "${service}"' in content
    assert 'update_service qwen-asr-api http://127.0.0.1:8002 /ready' in content
    assert "scripts/smoke_asr.sh" in content


def test_update_service_uses_asr_readiness_without_changing_other_health_checks():
    content = script("update_service.sh")

    assert 'if [[ "${SERVICE}" == "qwen-asr-api" ]]' in content
    assert 'CHECK_PATH="/ready"' in content
    assert 'CHECK_PATH="/health"' in content
    assert 'compose up -d --force-recreate --no-deps "${SERVICE}"' in content
    assert "scripts/smoke_asr.sh" in content


def test_remote_deploy_targets_asr_and_runs_websocket_smoke():
    content = script("deploy_remote.sh")

    assert 'SERVICE="${SERVICE:-qwen-asr-api}"' in content
    assert 'BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"' in content
    assert "scripts/smoke_asr.sh" in content


def test_asr_smoke_always_exercises_websocket_lifecycle():
    content = script("smoke_asr.sh")

    assert 'WebSocket smoke: start -> ready -> end -> final' in content
    assert '"type": "start"' in content
    assert '"type": "end"' in content
    assert 'expected ready' in content
    assert 'expected final' in content
    websocket_position = content.index("WebSocket smoke: start -> ready -> end -> final")
    audio_guard_position = content.index('if [[ -z "${AUDIO_FILE}" ]]')
    assert websocket_position < audio_guard_position


def test_a10_runbook_blocks_unvalidated_three_service_gpu_colocation():
    content = Path("cloud/README-A10.md").read_text()

    assert "Do not treat three-service GPU colocation as validated" in content
    assert "ASR_VLLM_GPU_MEMORY_UTILIZATION" in content
    assert "nvidia-smi" in content
