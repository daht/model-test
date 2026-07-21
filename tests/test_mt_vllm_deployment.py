from pathlib import Path

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
