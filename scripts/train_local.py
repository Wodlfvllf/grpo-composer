#!/usr/bin/env python3
"""Local single-node launcher (official fallback to Modal launcher).

This mirrors the core override behavior of scripts/train_modal.py without Modal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from grpo_composer.config.sanity import run_preflight_sanity_checks
from grpo_composer.runtime_stack import runtime_summary_text, validate_runtime_stack


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/tmp/grpo_data")
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


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
    overrides = _composer_yaml_to_overrides(config_path)
    extra_override_items = shlex.split(extra_overrides) if extra_overrides.strip() else []
    existing_keys = _extract_override_keys(overrides + extra_override_items)

    checkpoint_dir = CHECKPOINT_ROOT / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            "++trainer.project_name=grpo_composer_local",
            f"++trainer.experiment_name={_hydra_literal(run_name)}",
            "++trainer.resume_mode=disable",
        ]
    )

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
    if "actor_rollout_ref.model.external_lib" not in existing_keys:
        overrides.append("++actor_rollout_ref.model.external_lib=grpo_composer.integrations.verl")
    if "actor_rollout_ref.rollout.max_model_len" not in existing_keys:
        overrides.append("++actor_rollout_ref.rollout.max_model_len=2048")
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

    return ["python", "scripts/train_grpo.py", *overrides]


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
    env.setdefault("VERL_LOG_LEVEL", "WARNING")
    env.setdefault("RAY_DEDUP_LOGS", "1")
    env.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)

    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
