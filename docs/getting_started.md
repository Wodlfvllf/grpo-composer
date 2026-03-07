# Getting Started

## 1. Install

```bash
pip install -e .[verl]
```

## 2. Pick Launcher

### A. Canonical (Modal)

```bash
modal run scripts/train_modal.py \
  --config configs/base_grpo.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-preset gsm8k \
  --run-name grpo_modal_smoke \
  --total-epochs 1 \
  --n-gpus-per-node 1
```

### B. Local fallback (single node)

```bash
python scripts/train_local.py \
  --config configs/base_grpo.yaml \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-preset gsm8k \
  --run-name grpo_local_smoke \
  --total-epochs 1 \
  --n-gpus-per-node 1
```

## 3. What Happens Before Training

Both launchers enforce:

- runtime stack validation (`grpo_composer/runtime_stack.py`)
- config/data preflight sanity checks (`grpo_composer/config/sanity.py`)

If either fails, training exits immediately with actionable errors.

## 4. Checkpoints

- Modal default: `/checkpoints/<run_name>`
- Local default: `./checkpoints/<run_name>`

## 5. Common Misconfigurations Caught Early

- missing `actor_rollout_ref.actor.policy_loss.loss_mode=composer`
- unsupported `group_learnable + lambda_learnable=true`
- missing required parquet columns for special estimators/reward pipelines
- unsupported Info-GRPO path without augmented rollout tensors
