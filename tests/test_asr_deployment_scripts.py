import base64
import hashlib
import json
import os
import socketserver
import struct
import subprocess
import threading
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

    assert 'WebSocket smoke: start -> ready -> audio -> finish -> final -> close' in content
    assert '"type": "start"' in content
    assert '"type": "finish"' in content
    assert "math.sin" in content
    assert "send_frame(connection, 2" in content
    assert 'expected ready' in content
    assert 'expected final' in content
    websocket_position = content.index(
        "WebSocket smoke: start -> ready -> audio -> finish -> final -> close"
    )
    audio_guard_position = content.index('if [[ -z "${AUDIO_FILE}" ]]')
    assert websocket_position < audio_guard_position


def test_a10_runbook_blocks_unvalidated_three_service_gpu_colocation():
    content = Path("cloud/README-A10.md").read_text()

    assert "Do not treat three-service GPU colocation as validated" in content
    assert "ASR_VLLM_GPU_MEMORY_UTILIZATION" in content
    assert "nvidia-smi" in content


def test_production_examples_fail_closed_on_secrets_and_model_provenance():
    generic_env = Path(".env.example").read_text()
    a10_env = Path("cloud/A10.env.example").read_text()
    runbook = Path("cloud/README-A10.md").read_text()

    for example in (generic_env, a10_env):
        assert "API_KEY=\n" in example
        assert "ASR_REQUIRE_MODEL_MANIFEST=true" in example
        assert (
            "ASR_MODEL_MANIFEST_PATH=/models/Qwen3-ASR-1.7B.manifest.json"
            in example
        )
    assert "APPROVED_QWEN_REVISION" in runbook
    assert "external deployment gate" in runbook
    assert "unverified target-host download" in runbook


def _read_websocket_frame(stream):
    first, second = stream.read(2)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", stream.read(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", stream.read(8))[0]
    mask = stream.read(4) if second & 0x80 else b""
    payload = stream.read(length)
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return first & 0x0F, payload


def _send_websocket_json(stream, payload):
    encoded = json.dumps(payload).encode()
    stream.write(bytes([0x81, len(encoded)]) + encoded)
    stream.flush()


def _run_sequence_smoke(ready_sequence, final_sequence):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request_line = self.rfile.readline().decode()
            headers = {}
            while True:
                line = self.rfile.readline().decode()
                if line in {"\r\n", ""}:
                    break
                name, value = line.split(":", 1)
                headers[name.lower()] = value.strip()
            path = request_line.split()[1]
            if headers.get("upgrade", "").lower() == "websocket":
                accept = base64.b64encode(
                    hashlib.sha1(
                        (
                            headers["sec-websocket-key"]
                            + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                        ).encode()
                    ).digest()
                ).decode()
                self.wfile.write(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode()
                )
                self.wfile.flush()
                _read_websocket_frame(self.rfile)
                _send_websocket_json(
                    self.wfile, {"type": "ready", "sequence": ready_sequence}
                )
                _read_websocket_frame(self.rfile)
                _read_websocket_frame(self.rfile)
                _read_websocket_frame(self.rfile)
                _send_websocket_json(
                    self.wfile,
                    {"type": "final", "sequence": final_sequence, "text": ""},
                )
                self.wfile.write(b"\x88\x02\x03\xe8")
                self.wfile.flush()
                return
            if path == "/v1/transcribe/stream-info":
                body = {
                    "protocol_version": 2,
                    "file_transcribe_enabled": False,
                    "audio_format": {
                        "backend": "mock",
                        "stream_mode": "stateful",
                        "commit_on_punctuation": False,
                        "stateful": {"stable_commit_enabled": True},
                    },
                }
            else:
                body = {"status": "ready" if path == "/ready" else "ok"}
            encoded = json.dumps(body).encode()
            self.wfile.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(encoded)}\r\nConnection: close\r\n\r\n".encode()
                + encoded
            )
            self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = os.environ.copy()
        env.update(
            {
                "BASE_URL": f"http://127.0.0.1:{server.server_address[1]}",
                "API_KEY": "test-key",
                "EXPECT_ASR_BACKEND": "mock",
                "EXPECT_ASR_STREAM_MODE": "stateful",
            }
        )
        completed = subprocess.run(
            ["bash", "scripts/smoke_asr.sh"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
        )
        server.shutdown()
        thread.join(1)
        return completed


def test_asr_smoke_rejects_non_increasing_websocket_sequences():
    completed = _run_sequence_smoke(ready_sequence=99, final_sequence=1)

    assert completed.returncode != 0
    assert "sequence" in completed.stdout + completed.stderr


def test_asr_smoke_rejects_non_positive_websocket_sequence():
    completed = _run_sequence_smoke(ready_sequence=0, final_sequence=1)

    assert completed.returncode != 0
    assert "sequence" in completed.stdout + completed.stderr


def test_asr_smoke_requires_ready_sequence_to_start_at_one():
    completed = _run_sequence_smoke(ready_sequence=99, final_sequence=100)

    assert completed.returncode != 0
    assert "sequence" in completed.stdout + completed.stderr


def test_asr_smoke_rejects_websocket_sequence_gaps():
    completed = _run_sequence_smoke(ready_sequence=1, final_sequence=3)

    assert completed.returncode != 0
    assert "sequence" in completed.stdout + completed.stderr


def test_asr_runtime_and_silero_asset_supply_chain_are_fully_pinned():
    requirements = Path("requirements-asr-vllm.txt").read_text()
    dockerfile = Path("Dockerfile.asr").read_text()
    license_text = Path("licenses/SILERO-VAD-LICENSE").read_text()

    assert "qwen-asr[vllm]==0.0.6" in requirements
    assert "vllm==0.14.0" in requirements
    assert "onnxruntime==1.23.2" in requirements
    assert "SILERO_VAD_VERSION=6.2.1" in dockerfile
    assert "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert "SILERO-VAD-LICENSE" in dockerfile
    assert "MIT License" in license_text


def test_asr_image_uses_the_pinned_torch_and_cuda_runtime_base():
    dockerfile = Path("Dockerfile.asr").read_text()

    assert "FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime" in dockerfile


def test_asr_dependency_install_uses_a_reusable_buildkit_pip_cache():
    dockerfile = Path("Dockerfile.asr").read_text()

    assert "# syntax=docker/dockerfile:1.7" in dockerfile
    assert "RUN --mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "PIP_NO_CACHE_DIR" not in dockerfile


def test_asr_image_executes_official_qwen_streaming_contract_check():
    dockerfile = Path("Dockerfile.asr").read_text()
    contract = Path("scripts/check_qwen_streaming_contract.py").read_text()

    assert "check_qwen_streaming_contract.py" in dockerfile
    assert "ASRStreamingState" in contract
    for field in (
        "chunk_size_samples",
        "chunk_id",
        "buffer",
        "audio_accum",
        "language",
        "text",
    ):
        assert f'"{field}"' in contract
    assert 'require_version("qwen-asr", "0.0.6")' in contract
    assert 'require_version("vllm", "0.14.0")' in contract
def test_semantic_gateway_replaces_multipod_proxy_contract():
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text()
    readme = (root / "README.md").read_text()
    cloud = (root / "cloud" / "README-A10.md").read_text()

    assert not (root / "docker-compose.asr-multipod.yml").exists()
    assert not (root / "docs" / "asr-multipod-gateway.md").exists()
    assert "app.asr_gateway:app" in compose
    assert compose.count("--workers") == 1
    assert "ASR_GATEWAY_SCHEDULE_MAX_WAIT_MS" in readme
    assert "drain-and-switch" in readme
    assert "one model owner" in cloud
    assert "A10 capacity remains unverified" in cloud
    assert "environment" in readme.lower() and "credential" in readme.lower()


def test_shipped_clients_use_upgrade_auth_authenticated_info_and_finish_command():
    client = script("stream_asr_client.py")
    smoke = script("smoke_asr.sh")
    assert 'additional_headers={"X-API-Key": args.api_key}' in client
    assert 'fetch_stream_info(stream_info_url, args.api_key)' in client
    assert 'json.dumps({"type": "finish"})' in client
    assert '"api_key": args.api_key' not in client
    assert 'curl -fsS -H "X-API-Key: ${API_KEY}"' in smoke
    assert 'f"X-API-Key: {os.environ[\'API_KEY\']}\\r\\n' in smoke
    assert 'json.dumps({"type": "finish"})' in smoke
    assert '"type": "end"' not in smoke


def test_asr_api_docs_match_websocket_auth_completion_and_audio_error_contract():
    docs = Path("docs/API.md").read_text()
    streaming = docs.split("## ASR WebSocket Streaming", 1)[1].split(
        "## TTS Synthesis", 1
    )[0]
    python_example = docs.split("### ASR WebSocket", 1)[1].split(
        "## JavaScript Examples", 1
    )[0]
    browser_example = docs.split("## JavaScript Examples", 1)[1].split(
        "### TTS HTTP", 1
    )[0]
    start_message = streaming.split(
        "the authenticated upgrade must be JSON:", 1
    )[1].split("Server ready response:", 1)[0]
    error_contract = streaming.split("Every version 2 event", 1)[1].split(
        "The server limits active streams", 1
    )[0]
    asr_websocket_docs = streaming + python_example + browser_example

    assert "Authenticate the WebSocket upgrade with `X-API-Key`" in streaming
    assert "api_key" not in start_message
    assert '"api_key"' not in asr_websocket_docs
    assert '"type": "finish"' in streaming
    assert '"type": "end"' not in asr_websocket_docs
    assert "`invalid_audio` and closes the connection with code 1008" in streaming
    assert (
        "Gateway error codes emitted by the current Compose service are "
        "`invalid_start`, `invalid_audio`, `audio_limit`, `invalid_command`, "
        "`idle_timeout`, `session_timeout`, `audio_lag`, `backend_error`, "
        "`result_conflict`, and `overloaded`"
    ) in error_contract
    assert (
        "`invalid_start`, `invalid_audio`, `audio_limit`, and `invalid_command` "
        "close with 1008"
    ) in error_contract
    assert (
        "`idle_timeout`, `session_timeout`, `audio_lag`, `backend_error`, and "
        "`result_conflict` close with 1011"
    ) in error_contract
    assert "`overloaded` closes with 1013" in error_contract
    assert "1003" not in error_contract
    for unsupported_gateway_code in (
        "invalid_language",
        "frame_too_large",
        "server_busy",
        "realtime_lag_exceeded",
        "inference_timeout",
    ):
        assert unsupported_gateway_code not in error_contract
    assert 'new WebSocket("wss://asr-api.example.com' not in browser_example
    for browser_auth_option in (
        "trusted authentication proxy",
        "WebSocket subprotocol",
        "short-lived token",
        "translated to `X-API-Key`",
    ):
        assert browser_auth_option in browser_example
