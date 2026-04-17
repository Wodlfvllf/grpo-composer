from __future__ import annotations

from pathlib import Path

from launcher_common import (
    build_launcher_env,
    build_train_grpo_command,
    build_training_overrides,
)

from prepare_dataset import _prepare_dataset


def prepare_dataset(train_files: str, val_files: str, dataset_preset: str):
    return _prepare_dataset(train_files, val_files, dataset_preset)


def build_training_config(
    *,
    config_path: Path,
    model: str,
    train_files: str,
    val_files: str,
    run_name: str,
    total_epochs: int,
    n_gpus_per_node: int,
    checkpoint_dir: Path,
    extra_overrides: str,
    wandb_enabled: bool,
):
    default_logger = ["console", "wandb"] if wandb_enabled else ["console"]

    return build_training_overrides(
        config_path=config_path,
        model=model,
        train_files=train_files,
        val_files=val_files,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        project_name="grpo_composer_modal",
        default_logger=default_logger,
        extra_overrides=extra_overrides,
    )


def build_command(overrides):
    return build_train_grpo_command(overrides)


def build_env(remote_root):
    return build_launcher_env(remote_root)