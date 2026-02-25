"""
train_grpo.py — veRL GRPO training with grpo_composer components

This is the entry point that:
1. Imports grpo_composer.integrations.verl (triggers registration of all
   custom advantage estimators + the "composer" loss into veRL's registries)
2. Launches veRL's GRPO training loop with the specified config

Usage:
    # Single GPU (for debugging):
    python scripts/train_grpo.py --config configs/base_grpo.yaml --model.path Qwen/Qwen2.5-1.5B

    # Multi-GPU via torchrun:
    torchrun --nproc_per_node=4 scripts/train_grpo.py --config configs/dapo_7b.yaml

    # Or via the train.sh wrapper:
    bash scripts/train.sh configs/custom_mix.yaml Qwen/Qwen2.5-7B openai/gsm8k 4
"""

import os
import sys

# ────────────────────────────────────────────────────
# Step 1: Register grpo_composer components into veRL
# ────────────────────────────────────────────────────
# This single import triggers all @register_adv_est and @register_policy_loss
# decorators, making our custom components available to veRL.

import grpo_composer.integrations.verl  # noqa: F401  — side-effect import

# After this import, veRL's registries now contain:
#   ADV_ESTIMATOR_REGISTRY:
#     "difficulty_aware_grpo"   (GRPO-LEAD)
#     "length_corrected_grpo"   (TIC-GRPO)
#     "kalman_grpo"             (KRPO)
#     "decoupled_grpo"          (GDPO)
#     "multi_scale_grpo"        (MS-GRPO)
#     "static_value_grpo"       (PVPO)
#     "novelty_sharp_grpo"      (XRPO)
#
#   POLICY_LOSS_REGISTRY:
#     "composer"  — composable loss with configurable clip/agg/reg


# ────────────────────────────────────────────────────
# Step 2: Launch veRL's GRPO training loop
# ────────────────────────────────────────────────────

from verl.trainer.main_ppo import main


if __name__ == "__main__":
    print("=" * 60)
    print("  grpo_composer components registered into veRL")
    print("=" * 60)
    print()
    print("  Advantages:  difficulty_aware, length_corrected, kalman,")
    print("               decoupled, multi_scale, static_value, novelty_sharp")
    print("  Loss:        composer (clip_mode × agg_mode × regularizer)")
    print()
    print("=" * 60)

    main()
