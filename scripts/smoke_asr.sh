#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
API_KEY="${API_KEY:?Set API_KEY to the same value used in .env}"
AUDIO_FILE="${AUDIO_FILE:-}"
LANGUAGE="${LANGUAGE:-}"

echo "ASR health:"
curl -fsS "${BASE_URL}/health"
echo

echo "ASR readiness:"
curl -fsS "${BASE_URL}/ready"
echo

echo "ASR stream info:"
stream_info="$(curl -fsS "${BASE_URL}/v1/transcribe/stream-info")"
echo "${stream_info}"
echo

EXPECT_ASR_STREAM_MODE="${EXPECT_ASR_STREAM_MODE:-}"
EXPECT_ASR_BACKEND="${EXPECT_ASR_BACKEND:-}"
EXPECT_ASR_COMMIT_ON_PUNCTUATION="${EXPECT_ASR_COMMIT_ON_PUNCTUATION:-}"
EXPECT_ASR_STABLE_COMMIT_ENABLED="${EXPECT_ASR_STABLE_COMMIT_ENABLED:-}"
EXPECT_ASR_PROTOCOL_VERSION="${EXPECT_ASR_PROTOCOL_VERSION:-2}"

if [[ -n "${EXPECT_ASR_STREAM_MODE}${EXPECT_ASR_BACKEND}${EXPECT_ASR_COMMIT_ON_PUNCTUATION}${EXPECT_ASR_STABLE_COMMIT_ENABLED}${EXPECT_ASR_PROTOCOL_VERSION}" ]]; then
  STREAM_INFO_JSON="${stream_info}" \
  EXPECT_ASR_STREAM_MODE="${EXPECT_ASR_STREAM_MODE}" \
  EXPECT_ASR_BACKEND="${EXPECT_ASR_BACKEND}" \
  EXPECT_ASR_COMMIT_ON_PUNCTUATION="${EXPECT_ASR_COMMIT_ON_PUNCTUATION}" \
  EXPECT_ASR_STABLE_COMMIT_ENABLED="${EXPECT_ASR_STABLE_COMMIT_ENABLED}" \
  EXPECT_ASR_PROTOCOL_VERSION="${EXPECT_ASR_PROTOCOL_VERSION}" \
  python3 - <<'PY'
import json
import os

info = json.loads(os.environ["STREAM_INFO_JSON"])
audio = info["audio_format"]
expected_protocol = os.environ.get("EXPECT_ASR_PROTOCOL_VERSION", "")
if expected_protocol and str(info.get("protocol_version")) != expected_protocol:
    raise SystemExit(
        f"protocol_version expected {expected_protocol!r}, got {info.get('protocol_version')!r}"
    )
checks = {
    "EXPECT_ASR_STREAM_MODE": "stream_mode",
    "EXPECT_ASR_BACKEND": "backend",
    "EXPECT_ASR_COMMIT_ON_PUNCTUATION": "commit_on_punctuation",
}
for env_name, key in checks.items():
    expected = os.environ.get(env_name, "")
    if expected == "":
        continue
    actual = audio.get(key)
    if isinstance(actual, bool):
        actual = str(actual).lower()
        expected = expected.lower()
    if str(actual) != expected:
        raise SystemExit(f"{key} expected {expected!r}, got {actual!r}")

expected_stable_commit = os.environ.get("EXPECT_ASR_STABLE_COMMIT_ENABLED", "")
if expected_stable_commit:
    actual_stable_commit = audio.get("stateful", {}).get("stable_commit_enabled")
    if isinstance(actual_stable_commit, bool):
        actual_stable_commit = str(actual_stable_commit).lower()
        expected_stable_commit = expected_stable_commit.lower()
    if str(actual_stable_commit) != expected_stable_commit:
        raise SystemExit(
            "stateful.stable_commit_enabled "
            f"expected {expected_stable_commit!r}, got {actual_stable_commit!r}"
        )
PY
fi

echo "WebSocket smoke: start -> ready -> audio -> end -> final -> close"
BASE_URL="${BASE_URL}" API_KEY="${API_KEY}" LANGUAGE="${LANGUAGE}" python3 - <<'PY'
import base64
import hashlib
import json
import math
import os
import socket
import ssl
import struct
from urllib.parse import urlsplit


def receive_exact(connection, length):
    data = b""
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise SystemExit("WebSocket closed before final")
        data += chunk
    return data


def send_frame(connection, opcode, payload):
    if isinstance(payload, str):
        payload = payload.encode()
    mask = os.urandom(4)
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.extend([0x80 | 126])
        header.extend(struct.pack("!H", length))
    else:
        header.extend([0x80 | 127])
        header.extend(struct.pack("!Q", length))
    header.extend(mask)
    connection.sendall(header + bytes(value ^ mask[index % 4] for index, value in enumerate(payload)))


def receive_frame(connection):
    first, second = receive_exact(connection, 2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", receive_exact(connection, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", receive_exact(connection, 8))[0]
    if second & 0x80:
        mask = receive_exact(connection, 4)
        payload = bytes(
            value ^ mask[index % 4]
            for index, value in enumerate(receive_exact(connection, length))
        )
    else:
        payload = receive_exact(connection, length)
    return opcode, payload


def require_sequence(event, previous_sequence):
    if not isinstance(event, dict):
        raise SystemExit(f"expected JSON object event, got {event!r}")
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise SystemExit(f"event sequence must be a positive integer, got {sequence!r}")
    expected_sequence = previous_sequence + 1
    if sequence != expected_sequence:
        raise SystemExit(
            "event sequence must be continuous: "
            f"expected={expected_sequence}, got={sequence}"
        )
    return sequence


base = urlsplit(os.environ["BASE_URL"])
secure = base.scheme == "https"
host = base.hostname or "127.0.0.1"
port = base.port or (443 if secure else 80)
path_prefix = base.path.rstrip("/")
path = f"{path_prefix}/v1/transcribe/stream"
connection = socket.create_connection((host, port), timeout=30)
if secure:
    connection = ssl.create_default_context().wrap_socket(connection, server_hostname=host)
key = base64.b64encode(os.urandom(16)).decode()
request = (
    f"GET {path} HTTP/1.1\r\n"
    f"Host: {host}:{port}\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {key}\r\n"
    "Sec-WebSocket-Version: 13\r\n\r\n"
)
connection.sendall(request.encode())
response = b""
while b"\r\n\r\n" not in response:
    response += receive_exact(connection, 1)
status_line = response.split(b"\r\n", 1)[0]
if b" 101 " not in status_line:
    raise SystemExit(f"WebSocket upgrade failed: {status_line.decode(errors='replace')}")
expected_accept = base64.b64encode(
    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
).decode()
if f"sec-websocket-accept: {expected_accept}".lower().encode() not in response.lower():
    raise SystemExit("WebSocket upgrade returned an invalid accept key")

send_frame(
    connection,
    1,
    json.dumps(
        {
            "type": "start",
            "api_key": os.environ["API_KEY"],
            "language": os.environ.get("LANGUAGE") or None,
            "sample_rate": 16000,
            "format": "pcm_s16le",
        }
    ),
)
opcode, payload = receive_frame(connection)
ready = json.loads(payload)
if opcode != 1 or ready.get("type") != "ready":
    raise SystemExit(f"expected ready, got {ready!r}")
last_sequence = require_sequence(ready, 0)

pcm = b"".join(
    struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * index / 16000)))
    for index in range(16000)
)
send_frame(connection, 2, pcm[:16000])
send_frame(connection, 2, pcm[16000:])
send_frame(connection, 1, json.dumps({"type": "end"}))
final_count = 0
while True:
    opcode, payload = receive_frame(connection)
    if opcode == 8:
        break
    if opcode == 9:
        send_frame(connection, 10, payload)
        continue
    if opcode != 1:
        continue
    event = json.loads(payload)
    last_sequence = require_sequence(event, last_sequence)
    if event.get("type") == "error":
        raise SystemExit(f"ASR WebSocket error: {event!r}")
    if event.get("type") == "final":
        final_count += 1
        if "text" not in event:
            raise SystemExit(f"expected final with text, got {event!r}")
        print(json.dumps(event, ensure_ascii=False))
if final_count != 1:
    raise SystemExit(f"expected exactly one final, got {final_count}")
connection.close()
PY

if [[ -z "${AUDIO_FILE}" ]]; then
  echo "AUDIO_FILE is not set; readiness and WebSocket smoke completed."
  echo "To test transcription: AUDIO_FILE=/path/to/audio.wav API_KEY=<key> scripts/smoke_asr.sh"
  exit 0
fi

if [[ ! -f "${AUDIO_FILE}" ]]; then
  echo "Audio file not found: ${AUDIO_FILE}"
  exit 1
fi

file_transcribe_enabled="$({ STREAM_INFO_JSON="${stream_info}" python3 - <<'PY'
import json
import os

print(str(json.loads(os.environ["STREAM_INFO_JSON"]).get("file_transcribe_enabled", False)).lower())
PY
} )"
if [[ "${file_transcribe_enabled}" != "true" ]]; then
  echo "File transcription is disabled on this live streaming instance. Use a dedicated batch ASR instance."
  exit 1
fi

args=(
  -fsS
  -X POST
  "${BASE_URL}/v1/transcribe"
  -H "X-API-Key: ${API_KEY}"
  -F "file=@${AUDIO_FILE}"
)

if [[ -n "${LANGUAGE}" ]]; then
  args+=(-F "language=${LANGUAGE}")
fi

echo "ASR transcribe:"
curl "${args[@]}"
echo
