#!/bin/bash
# LEGACY WRAPPER
#
# Official launchers:
#   1) scripts/train_modal.py  (canonical production launcher)
#   2) scripts/train_local.py  (official local fallback)
#
# This script is retained only for backward compatibility.

set -euo pipefail

CONFIG=${1:-configs/base_grpo.yaml}
MODEL=${2:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET_PRESET=${3:-gsm8k}
NUM_GPUS=${4:-1}

if [[ "$DATASET_PRESET" == "openai/gsm8k" ]]; then
    DATASET_PRESET="gsm8k"
fi

echo "[legacy] scripts/train.sh is deprecated."
echo "[legacy] Redirecting to scripts/train_local.py."

exec python scripts/train_local.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --dataset-preset "$DATASET_PRESET" \
    --n-gpus-per-node "$NUM_GPUS" \
    --run-name "legacy_$(basename "$CONFIG" .yaml)"
