#!/usr/bin/env bash
set -euo pipefail

PROFILE="mt-vllm"
VLLM_SERVICE="hy-mt-vllm"
GATEWAY_SERVICE="hy-mt-api"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-600}"
HEALTH_INTERVAL_SECONDS="${HEALTH_INTERVAL_SECONDS:-5}"
LOG_TAIL="${LOG_TAIL:-120}"

if [[ ! -f "docker-compose.yml" ]]; then
  echo "Run this script from the project directory that contains docker-compose.yml."
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy cloud/A10.mt-vllm.env.example to .env and set API_KEY first."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker compose version >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "Docker Compose is not available."
  exit 1
fi

compose() {
  "${DOCKER[@]}" compose --profile "${PROFILE}" "$@"
}

wait_for_vllm() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  echo "Waiting for ${VLLM_SERVICE} readiness..."
  while (( SECONDS < deadline )); do
    if compose exec -T "${VLLM_SERVICE}" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep "${HEALTH_INTERVAL_SECONDS}"
  done
  echo "${VLLM_SERVICE} did not become ready within ${HEALTH_TIMEOUT_SECONDS}s."
  compose logs --tail="${LOG_TAIL}" "${VLLM_SERVICE}"
  return 1
}

wait_for_gateway() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  echo "Waiting for ${GATEWAY_SERVICE} health..."
  while (( SECONDS < deadline )); do
    if curl -fsS "${BASE_URL}/health"; then
      echo
      return 0
    fi
    sleep "${HEALTH_INTERVAL_SECONDS}"
  done
  echo "${GATEWAY_SERVICE} did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s."
  compose logs --tail="${LOG_TAIL}" "${GATEWAY_SERVICE}" "${VLLM_SERVICE}"
  return 1
}

echo "Pulling pinned vLLM image..."
compose pull "${VLLM_SERVICE}"
echo "Starting vLLM backend..."
compose up -d "${VLLM_SERVICE}"
wait_for_vllm
echo "Building translation gateway..."
compose build "${GATEWAY_SERVICE}"
echo "Recreating translation gateway..."
compose up -d --force-recreate --no-deps "${GATEWAY_SERVICE}"
wait_for_gateway
compose ps "${GATEWAY_SERVICE}" "${VLLM_SERVICE}"
