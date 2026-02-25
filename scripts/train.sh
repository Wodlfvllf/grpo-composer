#!/bin/bash
# ──────────────────────────────────────────────────────
# train.sh — Single-node veRL training with grpo_composer
# ──────────────────────────────────────────────────────
#
# Usage:
#   bash scripts/train.sh configs/dapo_7b.yaml
#   bash scripts/train.sh configs/custom_mix.yaml
#
# Requirements:
#   - pip install verl grpo-composer
#   - NVIDIA GPUs with CUDA (4× A100 80GB recommended for 7B models)
#   - vLLM installed for rollout generation
# ──────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──
CONFIG=${1:-configs/base_grpo.yaml}
MODEL=${2:-Qwen/Qwen2.5-7B}
DATASET=${3:-openai/gsm8k}
NUM_GPUS=${4:-4}

echo "═══════════════════════════════════════════════"
echo "  grpo_composer + veRL Training"
echo "═══════════════════════════════════════════════"
echo "  Config:   $CONFIG"
echo "  Model:    $MODEL"
echo "  Dataset:  $DATASET"
echo "  GPUs:     $NUM_GPUS"
echo "═══════════════════════════════════════════════"

# ── Validate environment ──
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  [{i}] {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)')
"

# ── Run training ──
# torchrun handles distributed setup (FSDP data parallel across GPUs)
# The grpo_composer components are registered via the train_grpo.py script
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    scripts/train_grpo.py \
    --config $CONFIG \
    --model.path $MODEL \
    --data.path $DATASET \
    --trainer.total_epochs 3 \
    --trainer.save_freq 1 \
    --trainer.project_name grpo_composer \
    --trainer.experiment_name $(basename $CONFIG .yaml)
