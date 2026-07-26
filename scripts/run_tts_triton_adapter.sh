#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_IMAGE="${TTS_TRITON_IMAGE:-soar97/triton-cosyvoice:25.06}"
TRITON_CONTAINER="${TTS_TRITON_CONTAINER:-cosyvoice-triton-server}"
API_CONTAINER="${TTS_API_CONTAINER:-cosyvoice-tts-api}"
API_PORT="${TTS_API_PORT:-8003}"
TRITON_HTTP_PORT="${TTS_TRITON_HTTP_PORT:-18000}"
TRITON_GRPC_URL="${TTS_TRITON_URL:-127.0.0.1:18001}"
MODEL_NAME="${TTS_MODEL_NAME:-Fun-CosyVoice3-0.5B-2512}"
TRITON_MODEL_NAME="${TTS_TRITON_MODEL_NAME:-cosyvoice3}"
PROMPT_WAV="${TTS_PROMPT_WAV:-/workspace/CosyVoice/asset/zero_shot_prompt.wav}"
API_KEY_VALUE="${API_KEY:-}"
ENV_FILE="${TTS_API_ENV_FILE:-${ROOT_DIR}/.env}"

usage() {
  cat <<'EOF'
Usage: scripts/run_tts_triton_adapter.sh [--foreground]

Starts the FastAPI WebSocket adapter in a separate Docker container.
The Triton container must already be running and expose HTTP 18000 and gRPC 18001.

Environment overrides:
  API_KEY                    Optional override; otherwise read from .env
  TTS_API_ENV_FILE           Environment file (default: <repo>/.env)
  TTS_API_CONTAINER          Adapter container name
  TTS_API_PORT               Adapter WebSocket port (default: 8003)
  TTS_TRITON_CONTAINER       Triton container name
  TTS_TRITON_IMAGE           Adapter image (default: soar97/triton-cosyvoice:25.06)
  TTS_MODEL_NAME             Public model name
  TTS_TRITON_MODEL_NAME      Triton model name
  TTS_PROMPT_WAV             Prompt path inside adapter container
EOF
}

foreground=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--foreground" ]]; then
  foreground=true
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }
[[ -d "${ROOT_DIR}/app" ]] || { echo "app directory not found: ${ROOT_DIR}/app" >&2; exit 2; }
[[ -f "${ROOT_DIR}/requirements-tts-triton.txt" ]] || {
  echo "requirements-tts-triton.txt not found" >&2
  exit 2
}
if [[ -z "${API_KEY_VALUE}" && ! -f "${ENV_FILE}" ]]; then
  echo "API_KEY is not set and environment file does not exist: ${ENV_FILE}" >&2
  exit 2
fi
if [[ -z "${API_KEY_VALUE}" ]] && ! grep -Eq '^[[:space:]]*API_KEY[[:space:]]*=[[:space:]]*[^[:space:]]' "${ENV_FILE}"; then
  echo "API_KEY is missing or empty in ${ENV_FILE}" >&2
  exit 2
fi

if ! docker inspect "${TRITON_CONTAINER}" >/dev/null 2>&1; then
  echo "Triton container does not exist: ${TRITON_CONTAINER}" >&2
  exit 2
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "${TRITON_CONTAINER}")" != "true" ]]; then
  echo "Triton container is not running: ${TRITON_CONTAINER}" >&2
  exit 2
fi

ready=false
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "${ready}" != "true" ]]; then
  echo "Triton is not ready on HTTP port ${TRITON_HTTP_PORT}" >&2
  exit 2
fi

docker rm -f "${API_CONTAINER}" >/dev/null 2>&1 || true

env_args=()
if [[ -f "${ENV_FILE}" ]]; then
  env_args+=(--env-file "${ENV_FILE}")
fi
api_key_args=()
if [[ -n "${API_KEY_VALUE}" ]]; then
  api_key_args+=(-e "API_KEY=${API_KEY_VALUE}")
fi

run_args=(
  run -d
  --name "${API_CONTAINER}"
  --restart unless-stopped
  --net host
  "${env_args[@]}"
  -e "ASR_BACKEND=mock"
  -e "MODEL_BACKEND=mock"
  -e "TTS_BACKEND=triton"
  -e "TTS_MODEL_NAME=${MODEL_NAME}"
  -e "TTS_TRITON_URL=${TRITON_GRPC_URL}"
  -e "TTS_TRITON_MODEL_NAME=${TRITON_MODEL_NAME}"
  -e "TTS_PROMPT_WAV=${PROMPT_WAV}"
  "${api_key_args[@]}"
  -v "${ROOT_DIR}:/workspace/model-test:ro"
  -v "${ROOT_DIR}/CosyVoice:/workspace/CosyVoice:ro"
  "${TRITON_IMAGE}"
  bash -lc
  "set -e; pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple -r /workspace/model-test/requirements.txt -r /workspace/model-test/requirements-tts-triton.txt; cd /workspace/model-test; exec uvicorn app.tts_api:app --host 0.0.0.0 --port ${API_PORT}"
)

container_id="$(docker "${run_args[@]}" | tr -d '\n')"
echo "TTS adapter started: ${API_CONTAINER} (${container_id})"
echo "WebSocket endpoint: ws://<host>:${API_PORT}/v1/tts/stream"

if [[ "${foreground}" == "true" ]]; then
  exec docker logs -f "${API_CONTAINER}"
fi
