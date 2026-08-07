#!/usr/bin/env bash

set -euo pipefail

ROLE="${1:-}"
if [[ "${ROLE}" != "planner" && "${ROLE}" != "prompt" ]]; then
    echo "Usage: $0 {planner|prompt}" >&2
    echo "  planner: serve open-ea/ea-3b-tab on port 12333 as eval-agent" >&2
    echo "  prompt:  serve Qwen/Qwen2.5-3B-Instruct on port 12334 as prompt-agent" >&2
    exit 2
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "Error: vllm is not installed in the current environment." >&2
    exit 1
fi

if [[ "${ROLE}" == "planner" ]]; then
    DEFAULT_MODEL="open-ea/ea-3b-tab"
    DEFAULT_PORT="12333"
    DEFAULT_SERVED_NAME="eval-agent"
else
    DEFAULT_MODEL="Qwen/Qwen2.5-3B-Instruct"
    DEFAULT_PORT="12334"
    DEFAULT_SERVED_NAME="prompt-agent"
fi

MODEL_NAME="${OPEN_EA_MODEL:-${DEFAULT_MODEL}}"
PORT="${OPEN_EA_PORT:-${DEFAULT_PORT}}"
SERVED_NAME="${OPEN_EA_SERVED_NAME:-${DEFAULT_SERVED_NAME}}"
OPEN_EA_BIND_HOST="${OPEN_EA_HOST:-127.0.0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

echo "Serving ${ROLE} model ${MODEL_NAME} at http://${OPEN_EA_BIND_HOST}:${PORT}/v1"

exec vllm serve "${MODEL_NAME}" \
    --host "${OPEN_EA_BIND_HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code
