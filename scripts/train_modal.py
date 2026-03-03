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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import modal
import yaml


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
    # Keep vLLM pinned to the 0.11 line to avoid WorkerWrapper API
    # incompatibilities seen with newer releases.
    .pip_install(
        "verl>=0.4.0",
        "ray[default]>=2.40.0",
        "vllm>=0.11.0,<0.12.0",
        "transformers>=4.51.0",
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    )
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


def _extract_gsm8k_solution(answer: str) -> str:
    match = re.search(r"####\s*(-?[0-9\.,]+)", answer)
    if not match:
        raise ValueError(f"Cannot extract GSM8K solution from answer: {answer[:120]}...")
    return match.group(1).replace(",", "")


def _prepare_gsm8k_dataset(output_dir: Path) -> tuple[str, str]:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"

    if train_path.exists() and val_path.exists():
        return str(train_path), str(val_path)

    dataset = load_dataset("openai/gsm8k", "main")
    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question = example["question"] + " " + instruction
            solution = _extract_gsm8k_solution(example["answer"])
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }

        return process_fn

    train_dataset = dataset["train"].map(make_map_fn("train"), with_indices=True)
    val_dataset = dataset["test"].map(make_map_fn("test"), with_indices=True)
    train_dataset.to_parquet(str(train_path))
    val_dataset.to_parquet(str(val_path))
    return str(train_path), str(val_path)


def _prepare_dataset(train_files: str, val_files: str, dataset_preset: str) -> tuple[str, str]:
    if train_files and val_files:
        return train_files, val_files
    if train_files or val_files:
        raise ValueError("Provide both train_files and val_files, or provide neither and use dataset_preset.")

    preset = dataset_preset.strip().lower()
    if preset == "gsm8k":
        return _prepare_gsm8k_dataset(DATA_ROOT / "gsm8k")
    raise ValueError(f"Unsupported dataset_preset='{dataset_preset}'. Supported: gsm8k.")


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
) -> str:
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
    if "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1" not in existing_keys:
        overrides.append('++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="0"')

    if extra_override_items:
        overrides.extend(extra_override_items)

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
    # vLLM V1 can be unstable for some model/runtime combinations in smoke runs.
    # Users can override by setting VLLM_USE_V1=1 in Modal environment.
    env.setdefault("VLLM_USE_V1", "0")
    print("Runtime env:", {"VLLM_USE_V1": env.get("VLLM_USE_V1")})
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REMOTE_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(REMOTE_ROOT)
    )

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
) -> None:
    print(f"Modal app: {APP_NAME}")
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
    )
    print(f"Training finished. Checkpoints written to: {checkpoint_path}")
