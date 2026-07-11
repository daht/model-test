#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-hy-mt-api}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODE="${1:-build}"
PULL_CODE="${PULL_CODE:-auto}"
LOG_TAIL="${LOG_TAIL:-120}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"
if [[ "${SERVICE}" == "qwen-asr-api" ]]; then
  CHECK_PATH="/ready"
else
  CHECK_PATH="/health"
fi

if [[ ! -f "docker-compose.yml" ]]; then
  echo "Run this script from the project directory that contains docker-compose.yml."
  echo "Example: cd /opt/model-test && scripts/update_service.sh"
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
    return
  fi

  if [[ ! -d ".git" ]]; then
    echo "No .git directory found; skipping git pull."
    return
  fi

  if [[ "${PULL_CODE}" == "auto" && -n "$(git status --porcelain)" ]]; then
    echo "Git working tree has local changes; skipping git pull."
    echo "Set PULL_CODE=1 to force git pull, or update files manually before running this script."
    return
  fi

  echo "Pulling latest code with git pull --ff-only..."
  git pull --ff-only
}

wait_for_health() {
  echo "Waiting for ${BASE_URL}${CHECK_PATH} ..."
  local deadline
  deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if curl -fsS "${BASE_URL}${CHECK_PATH}" >/tmp/service-health.json 2>/dev/null; then
      echo "Health check passed:"
      cat /tmp/service-health.json
      echo
      return
    fi
    sleep 5
  done

  echo "Health check failed after ${HEALTH_TIMEOUT_SECONDS}s."
  echo "Recent logs:"
  compose logs --tail="${LOG_TAIL}" "${SERVICE}"
  exit 1
}

show_status() {
  compose ps
  compose logs --tail="${LOG_TAIL}" "${SERVICE}"
}

run_asr_smoke() {
  if [[ "${SERVICE}" != "qwen-asr-api" ]]; then
    return
  fi
  echo "Running ASR WebSocket smoke..."
  (
    set -a
    source ./.env
    BASE_URL="${BASE_URL}" scripts/smoke_asr.sh
  )
}

case "${MODE}" in
  build)
    maybe_pull_code
    echo "Building ${SERVICE} while the old container keeps running..."
    compose build "${SERVICE}"
    echo "Recreating ${SERVICE}..."
    compose up -d --force-recreate --no-deps "${SERVICE}"
    wait_for_health
    run_asr_smoke
    show_status
    ;;
  env)
    echo "Recreating ${SERVICE} with current .env..."
    compose up -d --force-recreate --no-deps "${SERVICE}"
    wait_for_health
    run_asr_smoke
    show_status
    ;;
  restart)
    echo "Restarting ${SERVICE}..."
    compose restart "${SERVICE}"
    wait_for_health
    run_asr_smoke
    show_status
    ;;
  logs)
    compose logs -f --tail="${LOG_TAIL}" "${SERVICE}"
    ;;
  *)
    cat <<EOF
Usage:
  scripts/update_service.sh [build|env|restart|logs]

Modes:
  build    Pull code when safe, build image, recreate service. Default.
  env      Recreate service without rebuilding, for .env changes.
  restart  Restart service only, useful after replacing model files.
  logs     Follow service logs.

Environment:
  SERVICE=hy-mt-api
  BASE_URL=http://127.0.0.1:8000
  PULL_CODE=auto        # auto pulls only when safe; 0 disables git pull; 1 forces git pull
  LOG_TAIL=120
  HEALTH_TIMEOUT_SECONDS=300
EOF
    exit 2
    ;;
esac
