#!/bin/bash
# ──────────────────────────────────────────────────────
# train.sh — Single-node veRL training with grpo_composer
# ──────────────────────────────────────────────────────
#
# Usage:
#   bash scripts/train.sh configs/papers/dapo.yaml
#   bash scripts/train.sh configs/papers/custom_mix.yaml Qwen/Qwen2.5-7B openai/gsm8k 4
#
# Requirements:
#   - pip install grpo-composer[verl]
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
# veRL uses Hydra for config management. The grpo_composer YAML is
# flattened into Hydra ++key=value overrides.
python scripts/train_grpo.py \
    ++actor_rollout_ref.model.path="$MODEL" \
    ++data.path="$DATASET" \
    ++trainer.n_gpus_per_node="$NUM_GPUS" \
    ++trainer.nnodes=1 \
    ++trainer.total_epochs=3 \
    ++trainer.save_freq=1 \
    ++trainer.project_name=grpo_composer \
    ++trainer.experiment_name="$(basename "$CONFIG" .yaml)" \
    $(python -c "
import yaml, json, re, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f) or {}
def flatten(prefix, d, out):
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(f'{prefix}.{k}' if prefix else k, v, out)
    else:
        out[prefix] = d
flat = {}
flatten('', cfg, flat)
for k in sorted(flat):
    v = flat[k]
    if isinstance(v, bool):
        v = str(v).lower()
    elif isinstance(v, str):
        v = json.dumps(v)
    elif v is None:
        v = 'null'
    print(f'++{k}={v}')
")
