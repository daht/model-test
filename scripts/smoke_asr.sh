#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
API_KEY="${API_KEY:?Set API_KEY to the same value used in .env}"
AUDIO_FILE="${AUDIO_FILE:-}"
LANGUAGE="${LANGUAGE:-}"

echo "ASR health:"
curl -fsS "${BASE_URL}/health"
echo

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
