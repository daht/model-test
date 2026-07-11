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
                _send_websocket_json(
                    self.wfile,
                    {"type": "final", "sequence": final_sequence, "text": ""},
                )
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
