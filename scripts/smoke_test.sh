#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-}"
REMOTE_HOST="${REMOTE_HOST:-}"
API_KEY="${API_KEY:?Set API_KEY to the same value used in .env}"

if [[ -z "${BASE_URL}" ]]; then
  if [[ -z "${REMOTE_HOST}" ]]; then
    echo "Set BASE_URL, for example http://1.2.3.4:8000, or set REMOTE_HOST."
    exit 1
  fi
  HOST="${REMOTE_HOST#*@}"
  BASE_URL="http://${HOST}:8000"
fi

echo "Health:"
curl -fsS "${BASE_URL}/health"
echo

echo "Translate:"
curl -fsS -X POST "${BASE_URL}/v1/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{
    "source_lang": "zh",
    "target_lang": "en",
    "text": "你好，欢迎使用我们的产品。",
    "glossary": {
      "产品": "product"
    },
    "preserve_format": true
  }'
echo
