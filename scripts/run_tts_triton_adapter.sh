#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_API_IMAGE="cosyvoice-tts-adapter:latest"
API_IMAGE="${TTS_API_IMAGE:-${DEFAULT_API_IMAGE}}"
TRITON_CONTAINER="${TTS_TRITON_CONTAINER:-cosyvoice-triton-server}"
API_CONTAINER="${TTS_API_CONTAINER:-cosyvoice-tts-api}"
API_PORT="${TTS_API_PORT:-8003}"
TRITON_HTTP_PORT="${TTS_TRITON_HTTP_PORT:-18000}"
TRITON_GRPC_URL="${TTS_TRITON_URL:-127.0.0.1:18001}"
BACKEND="${TTS_BACKEND:-}"
MODEL_NAME="${TTS_MODEL_NAME:-}"
TRITON_MODEL_NAME="${TTS_TRITON_MODEL_NAME:-cosyvoice3}"
PROMPT_WAV="${TTS_PROMPT_WAV:-}"
API_KEY_VALUE="${API_KEY:-}"
ENV_FILE="${TTS_API_ENV_FILE:-${ROOT_DIR}/.env}"
VLLM_OMNI_BASE_URL="${TTS_VLLM_OMNI_BASE_URL:-}"
VLLM_OMNI_MODEL="${TTS_VLLM_OMNI_MODEL:-}"
VLLM_OMNI_ROOT="${TTS_VLLM_OMNI_ROOT:-}"
VLLM_OMNI_BIN="${TTS_VLLM_OMNI_BIN:-}"
VLLM_OMNI_PORT="${TTS_VLLM_OMNI_PORT:-8091}"
VLLM_OMNI_LOG="${TTS_VLLM_OMNI_LOG:-/tmp/vllm-omni-qwen-tts.log}"
VLLM_OMNI_PID="${TTS_VLLM_OMNI_PID:-/tmp/vllm-omni-qwen-tts.pid}"
VLLM_OMNI_START_TIMEOUT="${TTS_VLLM_OMNI_START_TIMEOUT_SECONDS:-900}"
VLLM_OMNI_API_KEY_VALUE="${TTS_VLLM_OMNI_API_KEY:-}"

env_file_value() {
  local key="$1"
  [[ -f "${ENV_FILE}" ]] || return 0
  awk -F= -v key="${key}" '$1 ~ "^[[:space:]]*" key "[[:space:]]*$" {sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2); print $2; exit}' "${ENV_FILE}"
}

BACKEND="${BACKEND:-$(env_file_value TTS_BACKEND)}"
MODEL_NAME="${MODEL_NAME:-$(env_file_value TTS_MODEL_NAME)}"
BACKEND="${BACKEND:-triton}"
MODEL_NAME="${MODEL_NAME:-Fun-CosyVoice3-0.5B-2512}"
VLLM_OMNI_BASE_URL="${VLLM_OMNI_BASE_URL:-$(env_file_value TTS_VLLM_OMNI_BASE_URL)}"
VLLM_OMNI_MODEL="${VLLM_OMNI_MODEL:-$(env_file_value TTS_VLLM_OMNI_MODEL)}"
VLLM_OMNI_BASE_URL="${VLLM_OMNI_BASE_URL:-http://127.0.0.1:8091}"
VLLM_OMNI_MODEL="${VLLM_OMNI_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-$(env_file_value TTS_VLLM_OMNI_ROOT)}"
VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-$(env_file_value TTS_VLLM_OMNI_BIN)}"
VLLM_OMNI_ROOT="${VLLM_OMNI_ROOT:-/opt/vllm-omni}"
VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-vllm-omni}"
VLLM_OMNI_API_KEY_VALUE="${VLLM_OMNI_API_KEY_VALUE:-$(env_file_value TTS_VLLM_OMNI_API_KEY)}"
vllm_url_port="${VLLM_OMNI_BASE_URL##*:}"
vllm_url_port="${vllm_url_port%%/*}"
if [[ "${vllm_url_port}" =~ ^[0-9]+$ ]]; then
  VLLM_OMNI_PORT="${vllm_url_port}"
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_tts_triton_adapter.sh [--foreground]

Starts the FastAPI WebSocket adapter in a separate Docker container. The backend
is selected by TTS_BACKEND in the environment file.

Environment overrides:
  API_KEY                    Optional override; otherwise read from .env
  TTS_API_ENV_FILE           Environment file (default: <repo>/.env)
  TTS_API_CONTAINER          Adapter container name
  TTS_API_PORT               Adapter WebSocket port (default: 8003)
  TTS_TRITON_CONTAINER       Triton container name
  TTS_API_IMAGE              Adapter image (default: cosyvoice-tts-adapter:latest)
  TTS_BACKEND                Deployment backend: triton, qwen, or vllm_omni
  TTS_MODEL_NAME             Public model name
  TTS_TRITON_MODEL_NAME      Triton model name
  TTS_PROMPT_WAV             Prompt path inside adapter container
  TTS_VLLM_OMNI_ROOT         vLLM-Omni checkout (default: /opt/vllm-omni)
  TTS_VLLM_OMNI_BIN          vLLM-Omni executable (default: vllm-omni)
  TTS_VLLM_OMNI_LOG          vLLM-Omni log path (default: /tmp/vllm-omni-qwen-tts.log)
  TTS_VLLM_OMNI_START_TIMEOUT_SECONDS  Startup wait (default: 900)
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
docker info >/dev/null 2>&1 || { echo "Docker daemon is not reachable" >&2; exit 2; }
[[ -d "${ROOT_DIR}/app" ]] || { echo "app directory not found: ${ROOT_DIR}/app" >&2; exit 2; }
[[ -f "${ROOT_DIR}/requirements-tts-adapter.txt" ]] || {
  echo "requirements-tts-adapter.txt not found" >&2
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

if [[ "${BACKEND}" == "triton" ]] && ! docker inspect "${TRITON_CONTAINER}" >/dev/null 2>&1; then
  echo "Triton container does not exist: ${TRITON_CONTAINER}" >&2
  exit 2
fi
if [[ "${BACKEND}" == "triton" ]] && [[ "$(docker inspect --format '{{.State.Running}}' "${TRITON_CONTAINER}")" != "true" ]]; then
  echo "Triton container is not running: ${TRITON_CONTAINER}" >&2
  exit 2
fi

if [[ "${BACKEND}" == "triton" ]]; then
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
elif [[ "${BACKEND}" != "qwen" && "${BACKEND}" != "vllm_omni" ]]; then
  echo "Unsupported TTS_BACKEND: ${BACKEND}; expected triton, qwen, or vllm_omni" >&2
  exit 2
fi

if [[ "${BACKEND}" == "vllm_omni" ]]; then
  vllm_ready=false
  if curl -fsS --max-time 2 "${VLLM_OMNI_BASE_URL}/health" >/dev/null 2>&1; then
    vllm_ready=true
    echo "Reusing ready vLLM-Omni at ${VLLM_OMNI_BASE_URL}"
  fi

  if [[ "${vllm_ready}" != "true" ]]; then
    vllm_host="${VLLM_OMNI_BASE_URL#*://}"
    vllm_host="${vllm_host%%/*}"
    vllm_host="${vllm_host%%:*}"
    if [[ "${vllm_host}" == "127.0.0.1" || "${vllm_host}" == "localhost" || "${vllm_host}" == "::1" ]]; then
      if [[ -f "${VLLM_OMNI_PID}" ]] && kill -0 "$(cat "${VLLM_OMNI_PID}")" 2>/dev/null; then
        echo "vLLM-Omni is starting (pid $(cat "${VLLM_OMNI_PID}")); waiting..."
      else
        command -v "${VLLM_OMNI_BIN}" >/dev/null 2>&1 || {
          echo "${VLLM_OMNI_BIN} is required to auto-start vLLM-Omni" >&2
          exit 2
        }
        [[ -d "${VLLM_OMNI_ROOT}" ]] || {
          echo "vLLM-Omni root not found: ${VLLM_OMNI_ROOT}" >&2
          exit 2
        }
        deploy_config="${TTS_VLLM_OMNI_DEPLOY_CONFIG:-${VLLM_OMNI_ROOT}/vllm_omni/deploy/qwen3_tts.yaml}"
        [[ -f "${deploy_config}" ]] || {
          echo "vLLM-Omni deploy config not found: ${deploy_config}" >&2
          exit 2
        }
        vllm_args=(
          serve "${VLLM_OMNI_MODEL}"
          --deploy-config "${deploy_config}"
          --host 0.0.0.0
          --port "${VLLM_OMNI_PORT}"
          --gpu-memory-utilization "${TTS_VLLM_OMNI_GPU_MEMORY_UTILIZATION:-0.9}"
          --trust-remote-code
          --omni
        )
        if [[ -n "${VLLM_OMNI_API_KEY_VALUE}" ]]; then
          vllm_args+=(--api-key "${VLLM_OMNI_API_KEY_VALUE}")
        fi
        echo "Starting vLLM-Omni ${VLLM_OMNI_MODEL}; log: ${VLLM_OMNI_LOG}"
        (
          cd "${VLLM_OMNI_ROOT}"
          nohup "${VLLM_OMNI_BIN}" "${vllm_args[@]}" >"${VLLM_OMNI_LOG}" 2>&1 &
          echo $! >"${VLLM_OMNI_PID}"
        )
      fi
    else
      echo "Waiting for remote vLLM-Omni at ${VLLM_OMNI_BASE_URL}"
    fi
  fi

  if [[ "${vllm_ready}" != "true" ]]; then
    for _ in $(seq 1 "${VLLM_OMNI_START_TIMEOUT}"); do
      if curl -fsS --max-time 2 "${VLLM_OMNI_BASE_URL}/health" >/dev/null 2>&1; then
        vllm_ready=true
        break
      fi
      if [[ -f "${VLLM_OMNI_PID}" ]] && ! kill -0 "$(cat "${VLLM_OMNI_PID}")" 2>/dev/null; then
        echo "vLLM-Omni exited during startup; see ${VLLM_OMNI_LOG}" >&2
        tail -n 80 "${VLLM_OMNI_LOG}" >&2 2>/dev/null || true
        exit 1
      fi
      sleep 1
    done
  fi
  if [[ "${vllm_ready}" != "true" ]]; then
    echo "vLLM-Omni did not become ready: ${VLLM_OMNI_BASE_URL}/health" >&2
    exit 1
  fi
fi

if [[ -z "${TTS_API_IMAGE:-}" ]]; then
  # Docker reuses unchanged dependency layers, while rebuilding here ensures
  # Dockerfile/requirements fixes (for example the SoX runtime) reach deploys.
  echo "Ensuring TTS adapter image ${API_IMAGE} is up to date..."
  docker build \
    --file "${ROOT_DIR}/Dockerfile.tts-adapter" \
    --tag "${API_IMAGE}" \
    "${ROOT_DIR}"
fi

port_in_use=false
if command -v ss >/dev/null 2>&1; then
  ss_output="$(ss -ltn 2>/dev/null || true)"
  if awk '{print $4}' <<<"${ss_output}" | grep -Eq "(^|:)${API_PORT}$"; then
    port_in_use=true
  fi
elif command -v netstat >/dev/null 2>&1; then
  if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${API_PORT}$"; then
    port_in_use=true
  fi
elif command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    port_in_use=true
  fi
else
  echo "Warning: no ss, netstat, or lsof; cannot preflight API port ${API_PORT}" >&2
fi
if [[ "${port_in_use}" == "true" ]]; then
  echo "API port ${API_PORT} is already in use; stop the existing service or set TTS_API_PORT" >&2
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

cosyvoice_mount=()
if [[ -d "${ROOT_DIR}/CosyVoice" ]]; then
  cosyvoice_mount=(-v "${ROOT_DIR}/CosyVoice:/workspace/CosyVoice:ro")
fi

gpu_args=()
if [[ "${BACKEND}" == "qwen" ]]; then
  gpu_args=(--gpus all)
fi

run_args=(
  run -d
  --name "${API_CONTAINER}"
  --restart unless-stopped
  --net host
  "${gpu_args[@]}"
  "${env_args[@]}"
  -e "ASR_BACKEND=mock"
  -e "MODEL_BACKEND=mock"
  -e "TTS_BACKEND=${BACKEND}"
  -e "TTS_MODEL_NAME=${MODEL_NAME}"
  -e "TTS_TRITON_URL=${TRITON_GRPC_URL}"
  -e "TTS_TRITON_MODEL_NAME=${TRITON_MODEL_NAME}"
  "${api_key_args[@]}"
  -v "${ROOT_DIR}:/workspace/model-test:ro"
  "${cosyvoice_mount[@]}"
  --mount "type=volume,source=tts-adapter-pip-cache,target=/root/.cache/pip"
  --mount "type=volume,source=tts-model-cache,target=/root/.cache/huggingface"
  "${API_IMAGE}"
  sh -ec
  "cd /workspace/model-test; exec python -m uvicorn app.tts_api:app --host 0.0.0.0 --port ${API_PORT}"
)

if [[ -n "${PROMPT_WAV}" ]]; then
  run_args+=( -e "TTS_PROMPT_WAV=${PROMPT_WAV}" )
elif ! grep -Eq '^[[:space:]]*TTS_PROMPT_WAV[[:space:]]*=' "${ENV_FILE}" 2>/dev/null; then
  run_args+=( -e "TTS_PROMPT_WAV=/workspace/CosyVoice/asset/zero_shot_prompt.wav" )
fi

container_id="$(docker "${run_args[@]}" | tr -d '\n')"
echo "TTS adapter started: ${API_CONTAINER} (${container_id})"
echo "WebSocket endpoint: ws://<host>:${API_PORT}/v1/tts/stream"

api_ready=false
for _ in $(seq 1 90); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    api_ready=true
    break
  fi
  if [[ "$(docker inspect --format '{{.State.Running}}' "${API_CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
    break
  fi
  sleep 2
done
if [[ "${api_ready}" != "true" ]]; then
  echo "TTS adapter did not become ready on port ${API_PORT}" >&2
  docker ps -a --filter "name=^/${API_CONTAINER}$" >&2 || true
  docker logs --tail 80 "${API_CONTAINER}" >&2 || true
  exit 1
fi
echo "TTS adapter health: http://127.0.0.1:${API_PORT}/health"

if [[ "${foreground}" == "true" ]]; then
  exec docker logs -f "${API_CONTAINER}"
fi
