#!/usr/bin/env bash
set -euo pipefail

PULL_CODE="${PULL_CODE:-auto}"
LOG_TAIL="${LOG_TAIL:-120}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"

if [[ ! -f "docker-compose.yml" ]]; then
  echo "Run this script from the project directory that contains docker-compose.yml."
  echo "Example: cd /opt/model-test && scripts/update_all_services.sh"
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy cloud/A10.env.example or .env.example to .env first."
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
  "${DOCKER[@]}" compose "$@"
}

maybe_pull_code() {
  if [[ "${PULL_CODE}" == "0" || "${PULL_CODE}" == "false" ]]; then
    echo "Skipping git pull because PULL_CODE=${PULL_CODE}."
    return
  fi

  if [[ ! -d ".git" ]]; then
    echo "No .git directory found; skipping git pull."
    return
  fi

  if [[ "${PULL_CODE}" == "auto" && -n "$(git status --porcelain)" ]]; then
    echo "Git working tree has local changes; skipping git pull."
    echo "Set PULL_CODE=1 to force git pull, or commit/stash local changes first."
    return
  fi

  echo "Pulling latest code with git pull --ff-only..."
  git pull --ff-only
}

wait_for_health() {
  local service="$1"
  local base_url="$2"
  local output_file="/tmp/${service}-health.json"
  local deadline
  deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

  echo "Waiting for ${service}: ${base_url}/health ..."
  while (( SECONDS < deadline )); do
    if curl -fsS "${base_url}/health" >"${output_file}" 2>/dev/null; then
      echo "${service} health check passed:"
      cat "${output_file}"
      echo
      return
    fi
    sleep 5
  done

  echo "${service} health check failed after ${HEALTH_TIMEOUT_SECONDS}s."
  echo "Recent logs:"
  compose logs --tail="${LOG_TAIL}" "${service}"
  exit 1
}

update_service() {
  local service="$1"
  local base_url="$2"

  echo "Recreating ${service}..."
  compose up -d --no-deps "${service}"
  wait_for_health "${service}" "${base_url}"
}

maybe_pull_code

echo "Building shared image..."
compose build hy-mt-api

update_service hy-mt-api http://127.0.0.1:8000
update_service qwen-asr-api http://127.0.0.1:8002

echo "Service status:"
compose ps

echo "Recent hy-mt-api logs:"
compose logs --tail="${LOG_TAIL}" hy-mt-api

echo "Recent qwen-asr-api logs:"
compose logs --tail="${LOG_TAIL}" qwen-asr-api

echo "All services updated."
