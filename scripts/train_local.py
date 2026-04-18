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
#!/usr/bin/env python3
"""Local single-node launcher (official fallback to Modal launcher).

This mirrors the core override behavior of scripts/train_modal.py without Modal.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT_CANDIDATE = _SCRIPT_PATH.parents[1]
if str(_REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_CANDIDATE))

try:
    from grpo_composer.config.sanity import run_preflight_sanity_checks
except Exception:  # pragma: no cover - fallback for older repo states
    def run_preflight_sanity_checks(*, config_path: Path, overrides: list[str], train_file: str) -> list[str]:
        return []

from prepare_dataset import (
    _prepare_dataset
)
from runtime_stack import (
    CANONICAL_PIP_PACKAGES,
    runtime_summary_text,
    validate_runtime_stack,
)
from launcher_common import (
    build_launcher_env,
    build_train_grpo_command,
    build_training_overrides,
    pkg_version,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/tmp/grpo_data")
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

def build_command(
    *,
    config_path: Path,
    model: str,
    train_files: str,
    val_files: str,
    run_name: str,
    total_epochs: int,
    n_gpus_per_node: int,
    extra_overrides: str,
) -> list[str]:
    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = build_training_overrides(
        config_path=config_path,
        model=model,
        train_files=train_files,
        val_files=val_files,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        project_name="grpo_composer_local",
        default_logger=["console"],
        extra_overrides=extra_overrides,
    )

    warnings = run_preflight_sanity_checks(
        config_path=config_path,
        overrides=overrides,
        train_file=train_files,
    )
    for warning in warnings:
        print(f"[sanity-warning] {warning}")

    return build_train_grpo_command(overrides)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local single-node launcher for grpo_composer")
    parser.add_argument("--config", default="configs/base_grpo.yaml")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-files", default="")
    parser.add_argument("--val-files", default="")
    parser.add_argument("--dataset-preset", default="gsm8k")
    parser.add_argument("--run-name", default="grpo_local_run")
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=1)
    parser.add_argument("--extra-overrides", default="")
    return parser.parse_args()


def main() -> None:
    validate_runtime_stack()

    args = parse_args()
    print(f"Canonical runtime stack: {runtime_summary_text()}")

    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Composer config not found: {config_path}")

    train_files, val_files = _prepare_dataset(args.train_files, args.val_files, args.dataset_preset)

    command = build_command(
        config_path=config_path,
        model=args.model,
        train_files=train_files,
        val_files=val_files,
        run_name=args.run_name,
        total_epochs=args.total_epochs,
        n_gpus_per_node=args.n_gpus_per_node,
        extra_overrides=args.extra_overrides,
    )

    print(
        "Runtime versions:",
        {
            "verl": pkg_version("verl"),
            "vllm": pkg_version("vllm"),
            "ray": pkg_version("ray"),
            "transformers": pkg_version("transformers"),
            "torch": pkg_version("torch"),
        },
    )
    print("Launching training command:")
    print(" ".join(shlex.quote(item) for item in command))

    env = build_launcher_env(REPO_ROOT)

    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
