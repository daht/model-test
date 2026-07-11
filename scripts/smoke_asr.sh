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

if [[ -z "${AUDIO_FILE}" ]]; then
  echo "AUDIO_FILE is not set; health check completed."
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
