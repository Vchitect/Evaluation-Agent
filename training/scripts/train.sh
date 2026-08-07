#!/usr/bin/env bash

set -euo pipefail

EA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EA_REPO_ROOT="$(cd -- "${EA_SCRIPT_DIR}/../.." && pwd)"
EA_CONFIG_PATH="${EA_REPO_ROOT}/training/configs/ea3b_full_sft.yaml"
EA_DATA_PATH="${EA_REPO_ROOT}/training/data/evaluation_agent_cot_dataset_table.json"
EA_LLAMAFACTORY_COMMIT="f00742b078599a7e5295f17c02acfef1b525492d"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "Error: llamafactory-cli is not installed." >&2
    echo "Initialize and install the pinned LLaMA-Factory submodule at ${EA_LLAMAFACTORY_COMMIT}; see training/README.md." >&2
    exit 1
fi

if [[ ! -f "${EA_DATA_PATH}" ]]; then
    echo "Error: processed EA-CoT-10K data was not found at:" >&2
    echo "  ${EA_DATA_PATH}" >&2
    echo "See training/README.md for the expected artifact and schema." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export WANDB_DIR="${WANDB_DIR:-${EA_REPO_ROOT}/training/outputs/wandb}"

IFS=',' read -r -a EA_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#EA_GPU_IDS[@]} != 4 )); then
    echo "Warning: the canonical effective batch size of 64 assumes four GPUs;" >&2
    echo "CUDA_VISIBLE_DEVICES currently exposes ${#EA_GPU_IDS[@]} device(s)." >&2
fi

cd "${EA_REPO_ROOT}"

echo "Starting EA-3B full-parameter SFT"
echo "Config: ${EA_CONFIG_PATH}"
echo "Data:   ${EA_DATA_PATH}"
echo "GPUs:   ${CUDA_VISIBLE_DEVICES}"

exec llamafactory-cli train "${EA_CONFIG_PATH}"
