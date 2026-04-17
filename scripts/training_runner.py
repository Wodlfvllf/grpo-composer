from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from runtime_stack import (
    runtime_summary_text,
    validate_runtime_stack,
)

from launcher_common import pkg_version

from training_config import (
    prepare_dataset,
    build_training_config,
    build_command,
    build_env,
)


def log_runtime_versions():
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


def execute_training(command, env, cwd):
    print("Launching training command:")
    print(" ".join(shlex.quote(item) for item in command))

    subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=True,
    )


def run_training_pipeline(
    *,
    config: str,
    model: str,
    train_files: str,
    val_files: str,
    dataset_preset: str,
    run_name: str,
    total_epochs: int,
    n_gpus_per_node: int,
    extra_overrides: str,
    debug: bool,
    remote_root: Path,
    checkpoint_root: Path,
    wandb_enabled: bool,
):
    validate_runtime_stack()

    if not run_name:
        raise ValueError("run_name is required")

    config_path = (remote_root / config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- pipeline ----
    train_files, val_files = prepare_dataset(
        train_files, val_files, dataset_preset
    )

    overrides = build_training_config(
        config_path=config_path,
        model=model,
        train_files=train_files,
        val_files=val_files,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        extra_overrides=extra_overrides,
        wandb_enabled=wandb_enabled,
    )

    command = build_command(overrides)

    log_runtime_versions()

    env = build_env(remote_root)

    if debug:
        env["GRPO_COMPOSER_DEBUG"] = "1"

    execute_training(command, env, remote_root)

    return str(checkpoint_dir)