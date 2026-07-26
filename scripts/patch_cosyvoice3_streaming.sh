#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_DIR="${COSYVOICE_DIR:-${ROOT_DIR}/CosyVoice}"
MODEL_FILE="${COSYVOICE_DIR}/runtime/triton_trtllm/model_repo_cosyvoice3/cosyvoice3/1/model.py"

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "CosyVoice3 Triton model file not found: ${MODEL_FILE}" >&2
  exit 2
fi

if grep -Fq 'get("dynamic_chunk_strategy", "time_based")' "${MODEL_FILE}"; then
  echo "CosyVoice3 streaming strategy already set to time_based"
  exit 0
fi

if ! grep -Fq 'get("dynamic_chunk_strategy", "exponential")' "${MODEL_FILE}"; then
  echo "Unexpected CosyVoice3 chunk strategy implementation; refusing to patch" >&2
  exit 2
fi

sed -i 's/get("dynamic_chunk_strategy", "exponential")/get("dynamic_chunk_strategy", "time_based")/' "${MODEL_FILE}"
echo "CosyVoice3 streaming strategy set to time_based in ${MODEL_FILE}"
