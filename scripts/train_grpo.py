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

import sys
from pathlib import Path

# Ensure local repository root is first on sys.path when this file is executed
# as a script (e.g., in Modal), so imports resolve to the mounted workspace code.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────
# Step 1: Register grpo_composer components into veRL
# ────────────────────────────────────────────────────
# This single import triggers all @register_adv_est and @register_policy_loss
# decorators, making our custom components available to veRL.
# NOTE: The collective_rpc patch for vLLM 0.11.2 is applied at image build
#       time via scripts/patch_verl_collective_rpc.py (not here, since Ray
#       workers are separate processes).

import grpo_composer.integrations.verl  # noqa: F401  — side-effect import
from grpo_composer.integrations.verl import patch_verl_main_ppo


# ────────────────────────────────────────────────────
# Step 2: Launch veRL's GRPO training loop
# ────────────────────────────────────────────────────

patch_verl_main_ppo()
from verl.trainer.main_ppo import main


if __name__ == "__main__":
    print("=" * 60)
    print("  grpo_composer components registered into veRL")
    print("=" * 60)
    print()
    print("  Advantages:  difficulty_aware, length_corrected, kalman,")
    print("               decoupled, multi_scale, static_value, novelty_sharp")
    print("  Loss:        composer (clip_mode × agg_mode × regularizer)")
    print("  Trainer:     ComposerRayPPOTrainer (patched over RayPPOTrainer)")
    print()
    print("=" * 60)

    main()
