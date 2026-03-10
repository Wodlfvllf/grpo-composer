#!/usr/bin/env python3
"""
Run grpo_composer training on Modal.

Usage:
    # Use existing parquet files:
    modal run scripts/train_modal.py \
      --config configs/base_grpo.yaml \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --train-files /data/gsm8k/train.parquet \
      --val-files /data/gsm8k/test.parquet \
      --run-name grpo_modal_run \
      --total-epochs 3 \
      --n-gpus-per-node 1

    # Or auto-prepare GSM8K parquet in the Modal container:
    modal run scripts/train_modal.py \
      --config configs/base_grpo.yaml \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --dataset-preset gsm8k \
      --run-name grpo_modal_run \
      --total-epochs 3 \
      --n-gpus-per-node 1

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
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import modal
import yaml

# Modal may execute this file from /root/train_modal.py while the repository is
# mounted at /root/grpo_composer (or baked elsewhere by legacy image builder).
# Ensure repository roots are present on sys.path before local package imports.
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_PATH_CANDIDATES = [
    _SCRIPT_PATH.parents[1],                # normal local layout: repo/scripts/train_modal.py
    _SCRIPT_PATH.parent / "grpo_composer",  # modal copied script at /root/train_modal.py
    Path("/root/grpo_composer"),            # explicit modal mount path used in this script
    Path("/"),                              # legacy image-builder copy root
]
for _candidate in _REPO_PATH_CANDIDATES:
    if (_candidate / "grpo_composer").exists():
        candidate_str = str(_candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

from grpo_composer.config.sanity import run_preflight_sanity_checks
from grpo_composer.runtime_stack import (
    CANONICAL_PIP_PACKAGES,
    runtime_summary_text,
    validate_runtime_stack,
)

# prepare_dataset.py lives alongside train_modal.py in scripts/
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
# In Modal container, files are copied to /root/grpo_composer/scripts/
_remote_scripts = "/root/grpo_composer/scripts"
if _remote_scripts not in sys.path:
    sys.path.insert(0, _remote_scripts)

from prepare_dataset import _prepare_gsm8k_dataset, _prepare_light_eval_maths_dataset


APP_NAME = "grpo-composer-train"
REMOTE_ROOT = Path("/root/grpo_composer")
CHECKPOINT_ROOT = Path("/checkpoints")
DATA_ROOT = Path("/tmp/grpo_data")


def _parse_gpu_config(value: str) -> str | list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "A100-80GB"
    return parts


GPU_CONFIG = _parse_gpu_config(os.environ.get("MODAL_GPU", "A100-80GB"))
TIMEOUT_SEC = int(os.environ.get("MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))
MAX_RETRIES = int(os.environ.get("MODAL_MAX_RETRIES", "0"))
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
    # Keep runtime pins centralized in grpo_composer.runtime_stack.
    .pip_install(*CANONICAL_PIP_PACKAGES)
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


def _prepare_dataset(train_files: str, val_files: str, dataset_preset: str) -> tuple[str, str]:
    if train_files and val_files:
        return train_files, val_files
    if train_files or val_files:
        raise ValueError("Provide both train_files and val_files, or provide neither and use dataset_preset.")

    preset = dataset_preset.strip().lower()
    if preset == "gsm8k":
        return _prepare_gsm8k_dataset(DATA_ROOT / "gsm8k")
    if preset in ("math", "math-hard", "lighteval"):
        return _prepare_light_eval_maths_dataset(DATA_ROOT / "lighteval_math")
    raise ValueError(f"Unsupported dataset_preset='{dataset_preset}'. Supported: gsm8k, math.")


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


def _extract_override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for item in overrides:
        raw = item.strip()
        if not raw:
            continue
        while raw.startswith("+"):
            raw = raw[1:]
        if "=" not in raw:
            continue
        key, _ = raw.split("=", 1)
        if key:
            keys.add(key)
    return keys


def _has_any(keys: set[str], *candidates: str) -> bool:
    return any(candidate in keys for candidate in candidates)


def _pkg_version(pkg_name: str) -> str:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return "not-installed"


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT_SEC,
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
    secrets=_secrets,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    max_inputs=1,
)
def run_training(
    config: str,
    model: str,
    train_files: str,
    val_files: str,
    dataset_preset: str,
    run_name: str,
    total_epochs: int = 3,
    n_gpus_per_node: int = 1,
    extra_overrides: str = "",
    debug : bool = False,
) -> str:
    validate_runtime_stack()

    if not run_name:
        raise ValueError("run_name is required")

    config_path = (REMOTE_ROOT / config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Composer config not found: {config_path}")

    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = _prepare_dataset(train_files, val_files, dataset_preset)

    overrides = _composer_yaml_to_overrides(config_path)
    extra_override_items = shlex.split(extra_overrides) if extra_overrides.strip() else []
    existing_keys = _extract_override_keys(overrides + extra_override_items)
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

    # veRL requires explicit micro-batch fields when dynamic batch sizing is off.
    # Add conservative defaults unless user/config already set them.
    if not _has_any(
        existing_keys,
        "actor_rollout_ref.actor.ppo_micro_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8")
    if not _has_any(
        existing_keys,
        "actor_rollout_ref.ref.log_prob_micro_batch_size",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8")
    if not _has_any(
        existing_keys,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8")
    if not _has_any(
        existing_keys,
        "critic.forward_micro_batch_size",
        "critic.forward_micro_batch_size_per_gpu",
    ):
        overrides.append("++critic.forward_micro_batch_size_per_gpu=8")
    if not _has_any(
        existing_keys,
        "critic.ppo_micro_batch_size",
        "critic.ppo_micro_batch_size_per_gpu",
    ):
        overrides.append("++critic.ppo_micro_batch_size_per_gpu=8")
    if "actor_rollout_ref.rollout.tensor_model_parallel_size" not in existing_keys:
        overrides.append(f"++actor_rollout_ref.rollout.tensor_model_parallel_size={max(1, n_gpus_per_node)}")
    if "actor_rollout_ref.rollout.name" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.name=vllm")
    if "actor_rollout_ref.rollout.enforce_eager" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.enforce_eager=true")
    if "actor_rollout_ref.rollout.enable_prefix_caching" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.enable_prefix_caching=false")
    if "actor_rollout_ref.rollout.enable_chunked_prefill" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.enable_chunked_prefill=false")
    if "actor_rollout_ref.rollout.load_format" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.load_format=auto")
    # Register grpo_composer losses/advantages in FSDP worker processes.
    # veRL's external_lib calls importlib.import_module() in every worker.
    if "actor_rollout_ref.model.external_lib" not in existing_keys:
        overrides.append('++actor_rollout_ref.model.external_lib=grpo_composer.integrations.verl')
    # Limit max sequence length to avoid OOM in vLLM V1 EngineCore subprocess.
    # Default 32768 (Qwen2.5) exhausts KV cache with shared GPU memory.
    if "actor_rollout_ref.rollout.max_model_len" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.max_model_len=2048")
    # On single-GPU setups (FSDP + vLLM share the same GPU), limit vLLM memory
    if "actor_rollout_ref.rollout.gpu_memory_utilization" not in existing_keys and n_gpus_per_node <= 2:
        overrides.append("++actor_rollout_ref.rollout.gpu_memory_utilization=0.5")
    if "trainer.logger" not in existing_keys:
        overrides.append(f"++trainer.logger={_hydra_literal(['console'])}")
    if not _has_any(
        existing_keys,
        "actor_rollout_ref.model.override_config._attn_implementation",
        "actor_rollout_ref.model.override_config.attn_implementation",
    ):
        overrides.append("++actor_rollout_ref.model.override_config.attn_implementation=sdpa")
        overrides.append("++actor_rollout_ref.model.override_config._attn_implementation=sdpa")
    if not _has_any(
        existing_keys,
        "critic.model.override_config._attn_implementation",
        "critic.model.override_config.attn_implementation",
    ):
        overrides.append("++critic.model.override_config.attn_implementation=sdpa")
        overrides.append("++critic.model.override_config._attn_implementation=sdpa")

    if extra_override_items:
        overrides.extend(extra_override_items)

    warnings = run_preflight_sanity_checks(
        config_path=config_path,
        overrides=overrides,
        train_file=train_files,
    )
    for warning in warnings:
        print(f"[sanity-warning] {warning}")

    command = ["python", "scripts/train_grpo.py", *overrides]
    print(
        "Runtime versions:",
        {
            "verl": _pkg_version("verl"),
            "vllm": _pkg_version("vllm"),
            "ray": _pkg_version("ray"),
            "transformers": _pkg_version("transformers"),
            "torch": _pkg_version("torch"),
        },
    )
    print("Launching training command:")
    print(" ".join(shlex.quote(item) for item in command))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Suppress verbose config dumps and Ray noise — show only errors + progress
    env.setdefault("VERL_LOG_LEVEL", "WARNING")
    env.setdefault("RAY_DEDUP_LOGS", "1")
    env.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REMOTE_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(REMOTE_ROOT)
    )
    if debug:
        env["GRPO_COMPOSER_DEBUG"] = "1"

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
    dataset_preset: str = "gsm8k",
    run_name: str = "grpo_modal_run",
    total_epochs: int = 3,
    n_gpus_per_node: int = 1,
    extra_overrides: str = "",
    debug: bool = False,
) -> None:

    print(f"Modal app: {APP_NAME}")
    print(f"Canonical runtime stack: {runtime_summary_text()}")
    print(f"GPU config: {GPU_CONFIG}")
    print(f"Checkpoint volume: {CHECKPOINT_VOLUME_NAME}")
    if train_files and val_files:
        print(f"Dataset mode: explicit files ({train_files}, {val_files})")
    else:
        print(f"Dataset mode: preset={dataset_preset} (auto-prepared in container)")
    if _secrets:
        print(f"Attached secrets: {len(_secrets)}")
    else:
        print("Attached secrets: none")

    checkpoint_path = run_training.remote(
        config=config,
        model=model,
        train_files=train_files,
        val_files=val_files,
        dataset_preset=dataset_preset,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        extra_overrides=extra_overrides,
        debug=debug,
    )

    print(f"Training finished. Checkpoints written to: {checkpoint_path}")
