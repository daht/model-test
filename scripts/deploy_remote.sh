#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST, for example root@1.2.3.4}"
REMOTE_DIR="${REMOTE_DIR:-/opt/hy-mt-api}"
ENV_FILE="${ENV_FILE:-.env}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example to .env and edit it first."
  exit 1
fi

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".env" \
  --exclude "models" \
  ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

scp "${ENV_FILE}" "${REMOTE_HOST}:${REMOTE_DIR}/.env"

ssh "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && BASE_URL='${BASE_URL}' PULL_CODE=0 scripts/update_service.sh build"

echo "Deployment command completed."
echo "Run: REMOTE_HOST=${REMOTE_HOST} API_KEY=<key> scripts/smoke_test.sh"
