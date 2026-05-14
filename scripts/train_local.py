#!/usr/bin/env python3
"""Local single-node launcher for grpo-composer.

Mirrors ``scripts/train_modal.py`` but runs in the current Python process,
no Modal required. Both launchers share ``scripts.training_runner`` so
behaviour is identical (same env, same overrides, same DAPO-debug wiring).

Examples
--------
    python scripts/train_local.py --config configs/base_grpo.yaml
    python scripts/train_local.py --config examples/kalman_dapo.yaml --debug
    python scripts/train_local.py --config configs/papers/dapo.yaml \\
        --model Qwen/Qwen2.5-1.5B-Instruct --n-gpus-per-node 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training_runner import run_training_pipeline  # noqa: E402


CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local single-node launcher for grpo-composer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/base_grpo.yaml",
                   help="Composer YAML, relative to repo root")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-files", default="")
    p.add_argument("--val-files", default="")
    p.add_argument("--dataset-preset", default="gsm8k")
    p.add_argument("--run-name", default="grpo_local_run")
    p.add_argument("--total-epochs", type=int, default=1)
    p.add_argument("--n-gpus-per-node", type=int, default=1)
    p.add_argument("--extra-overrides", default="",
                   help="Extra Hydra overrides, space-separated, e.g. "
                        "'++data.train_batch_size=512 ++trainer.save_freq=10'")
    p.add_argument("--debug", action="store_true",
                   help="Enable composer + DAPO debug prints "
                        "(GRPO_COMPOSER_DEBUG, DAPO debug config flag, "
                        "GRPO_COMPOSER_STRICT_VALIDATION).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    run_training_pipeline(
        config=args.config,
        model=args.model,
        train_files=args.train_files,
        val_files=args.val_files,
        dataset_preset=args.dataset_preset,
        run_name=args.run_name,
        total_epochs=args.total_epochs,
        n_gpus_per_node=args.n_gpus_per_node,
        extra_overrides=args.extra_overrides,
        debug=args.debug,
        remote_root=REPO_ROOT,
        checkpoint_root=CHECKPOINT_ROOT,
        wandb_enabled=False,
    )


if __name__ == "__main__":
    main()
