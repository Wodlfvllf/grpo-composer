#!/bin/bash
# ──────────────────────────────────────────────────────
# train_multinode.sh — Multi-node veRL training with grpo_composer
# ──────────────────────────────────────────────────────
#
# Usage (run on EACH node):
#   NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train_multinode.sh configs/dapo_7b.yaml
#   NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train_multinode.sh configs/dapo_7b.yaml
#
# Requirements:
#   - Same as train.sh
#   - All nodes can reach MASTER_ADDR:MASTER_PORT
#   - Same Python environment on all nodes
#   - Same model/dataset paths accessible from all nodes (NFS or download)
# ──────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──
CONFIG=${1:-configs/base_grpo.yaml}
MODEL=${2:-Qwen/Qwen2.5-7B}
DATASET=${3:-openai/gsm8k}

# ── Multi-node settings (set via environment) ──
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

echo "═══════════════════════════════════════════════"
echo "  grpo_composer + veRL — Multi-Node Training"
echo "═══════════════════════════════════════════════"
echo "  Config:       $CONFIG"
echo "  Model:        $MODEL"
echo "  Nodes:        $NNODES (this is node $NODE_RANK)"
echo "  GPUs/node:    $GPUS_PER_NODE"
echo "  Total GPUs:   $(($NNODES * $GPUS_PER_NODE))"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "═══════════════════════════════════════════════"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_grpo.py \
    --config $CONFIG \
    --model.path $MODEL \
    --data.path $DATASET \
    --trainer.total_epochs 3 \
    --trainer.save_freq 1 \
    --trainer.project_name grpo_composer \
    --trainer.experiment_name $(basename $CONFIG .yaml)_${NNODES}node
