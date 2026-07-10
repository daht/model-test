#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
API_KEY="${API_KEY:?Set API_KEY to the same value used in .env}"
AUDIO_FILE="${AUDIO_FILE:-}"
LANGUAGE="${LANGUAGE:-}"

echo "ASR health:"
curl -fsS "${BASE_URL}/health"
echo

echo "ASR stream info:"
stream_info="$(curl -fsS "${BASE_URL}/v1/transcribe/stream-info")"
echo "${stream_info}"
echo

EXPECT_ASR_STREAM_MODE="${EXPECT_ASR_STREAM_MODE:-}"
EXPECT_ASR_BACKEND="${EXPECT_ASR_BACKEND:-}"
EXPECT_ASR_COMMIT_ON_PUNCTUATION="${EXPECT_ASR_COMMIT_ON_PUNCTUATION:-}"

if [[ -n "${EXPECT_ASR_STREAM_MODE}${EXPECT_ASR_BACKEND}${EXPECT_ASR_COMMIT_ON_PUNCTUATION}" ]]; then
  STREAM_INFO_JSON="${stream_info}" \
  EXPECT_ASR_STREAM_MODE="${EXPECT_ASR_STREAM_MODE}" \
  EXPECT_ASR_BACKEND="${EXPECT_ASR_BACKEND}" \
  EXPECT_ASR_COMMIT_ON_PUNCTUATION="${EXPECT_ASR_COMMIT_ON_PUNCTUATION}" \
  python3 - <<'PY'
import json
import os

info = json.loads(os.environ["STREAM_INFO_JSON"])
audio = info["audio_format"]
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
