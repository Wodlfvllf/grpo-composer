#!/usr/bin/env python3
"""
Run grpo_composer training on Modal.

Usage:
    modal run scripts/train_modal.py \
      --config configs/base_grpo.yaml \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --train-files /data/gsm8k/train.parquet \
      --val-files /data/gsm8k/test.parquet \
      --run-name grpo_modal_run \
      --total-epochs 3 \
      --n-gpus-per-node 1 \
      --extra-overrides "trainer.test_freq=2"

Environment variables:
    MODAL_GPU                  GPU config string or CSV fallback list.
                               Default: A100-80GB
                               Example: H100,A100-80GB,A100
    MODAL_TIMEOUT_SEC          Function timeout in seconds. Default: 86400
    MODAL_CHECKPOINT_VOLUME    Volume name for checkpoints.
                               Default: grpo-composer-checkpoints
    MODAL_HF_SECRET_NAME       Optional Modal Secret name for HF token.
    MODAL_WANDB_SECRET_NAME    Optional Modal Secret name for W&B creds.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import modal
import yaml


APP_NAME = "grpo-composer-train"
REMOTE_ROOT = Path("/root/grpo_composer")
CHECKPOINT_ROOT = Path("/checkpoints")


def _parse_gpu_config(value: str) -> str | list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "A100-80GB"
    return parts


GPU_CONFIG = _parse_gpu_config(os.environ.get("MODAL_GPU", "A100-80GB"))
TIMEOUT_SEC = int(os.environ.get("MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "MODAL_CHECKPOINT_VOLUME",
    "grpo-composer-checkpoints",
)
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "").strip()
WANDB_SECRET_NAME = os.environ.get("MODAL_WANDB_SECRET_NAME", "").strip()


def _get_secret(secret_name: str) -> modal.Secret | None:
    if not secret_name:
        return None
    return modal.Secret.from_name(secret_name)


_secrets = [
    secret
    for secret in (_get_secret(HF_SECRET_NAME), _get_secret(WANDB_SECRET_NAME))
    if secret is not None
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("verl>=0.4.0", "ray>=2.40.0", "vllm>=0.8.3")
    .add_local_dir(".", remote_path=str(REMOTE_ROOT), copy=True)
)

app = modal.App(APP_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


def _flatten_mapping(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _flatten_mapping(next_prefix, nested, out)
        return
    out[prefix] = value


_SCIENTIFIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")
_INTEGER_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+)$")


def _coerce_scalar_string(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if _INTEGER_RE.fullmatch(stripped):
        try:
            return int(stripped)
        except ValueError:
            return value
    if _FLOAT_RE.fullmatch(stripped) or _SCIENTIFIC_RE.fullmatch(stripped):
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def _hydra_literal(value: Any) -> str:
    if isinstance(value, str):
        value = _coerce_scalar_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _composer_yaml_to_overrides(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return []
    if not isinstance(loaded, dict):
        raise ValueError(f"Composer config must be a mapping, got {type(loaded)} at {config_path}")

    flat: dict[str, Any] = {}
    _flatten_mapping("", loaded, flat)

    overrides: list[str] = []
    for key in sorted(flat.keys()):
        if not key:
            continue
        overrides.append(f"++{key}={_hydra_literal(flat[key])}")
    return overrides


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT_SEC,
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
    secrets=_secrets,
    retries=modal.Retries(initial_delay=0.0, max_retries=2),
    max_inputs=1,
)
def run_training(
    config: str,
    model: str,
    train_files: str,
    val_files: str,
    run_name: str,
    total_epochs: int = 3,
    n_gpus_per_node: int = 1,
    extra_overrides: str = "",
) -> str:
    if not train_files:
        raise ValueError("train_files is required")
    if not val_files:
        raise ValueError("val_files is required")
    if not run_name:
        raise ValueError("run_name is required")

    config_path = (REMOTE_ROOT / config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Composer config not found: {config_path}")

    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overrides = _composer_yaml_to_overrides(config_path)
    overrides.extend(
        [
            f"++data.train_files={_hydra_literal(train_files)}",
            f"++data.val_files={_hydra_literal(val_files)}",
            f"++actor_rollout_ref.model.path={_hydra_literal(model)}",
            f"++critic.model.path={_hydra_literal(model)}",
            f"++trainer.n_gpus_per_node={n_gpus_per_node}",
            "++trainer.nnodes=1",
            "++trainer.val_before_train=false",
            f"++trainer.default_local_dir={_hydra_literal(str(checkpoint_dir))}",
            f"++trainer.total_epochs={total_epochs}",
            "++trainer.save_freq=1",
            "++trainer.project_name=grpo_composer_modal",
            f"++trainer.experiment_name={_hydra_literal(run_name)}",
            "++trainer.resume_mode=disable",
        ]
    )

    if extra_overrides.strip():
        overrides.extend(shlex.split(extra_overrides))

    command = ["python", "scripts/train_grpo.py", *overrides]
    print("Launching training command:")
    print(" ".join(shlex.quote(item) for item in command))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    subprocess.run(
        command,
        cwd=str(REMOTE_ROOT),
        env=env,
        check=True,
    )
    return str(checkpoint_dir)


@app.local_entrypoint()
def main(
    config: str = "configs/base_grpo.yaml",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_files: str = "",
    val_files: str = "",
    run_name: str = "grpo_modal_run",
    total_epochs: int = 3,
    n_gpus_per_node: int = 1,
    extra_overrides: str = "",
) -> None:
    if not train_files:
        raise ValueError("--train-files is required")
    if not val_files:
        raise ValueError("--val-files is required")

    print(f"Modal app: {APP_NAME}")
    print(f"GPU config: {GPU_CONFIG}")
    print(f"Checkpoint volume: {CHECKPOINT_VOLUME_NAME}")
    if _secrets:
        print(f"Attached secrets: {len(_secrets)}")
    else:
        print("Attached secrets: none")

    checkpoint_path = run_training.remote(
        config=config,
        model=model,
        train_files=train_files,
        val_files=val_files,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        extra_overrides=extra_overrides,
    )
    print(f"Training finished. Checkpoints written to: {checkpoint_path}")
