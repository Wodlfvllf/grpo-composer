"""Shared launcher utilities for local and Modal training entrypoints.

These helpers are intentionally pure or near-pure to make behavior easy to test.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


_SCIENTIFIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")
_INTEGER_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+)$")


def _flatten_mapping(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            _flatten_mapping(next_prefix, nested, out)
        return
    out[prefix] = value


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


def hydra_literal(value: Any) -> str:
    if isinstance(value, str):
        value = _coerce_scalar_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def composer_yaml_to_overrides(config_path: Path) -> list[str]:
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
        overrides.append(f"++{key}={hydra_literal(flat[key])}")
    return overrides


def parse_extra_overrides(extra_overrides: str) -> list[str]:
    return shlex.split(extra_overrides) if extra_overrides.strip() else []


def extract_override_keys(overrides: list[str]) -> set[str]:
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


def has_any(keys: set[str], *candidates: str) -> bool:
    return any(candidate in keys for candidate in candidates)


def build_training_overrides(
    *,
    config_path: Path,
    model: str,
    train_files: str,
    val_files: str,
    run_name: str,
    total_epochs: int,
    n_gpus_per_node: int,
    checkpoint_dir: Path,
    project_name: str,
    default_logger: list[str],
    extra_overrides: str,
) -> list[str]:
    overrides = composer_yaml_to_overrides(config_path)
    extra_override_items = parse_extra_overrides(extra_overrides)
    existing_keys = extract_override_keys(overrides + extra_override_items)

    overrides.extend(
        [
            f"++data.train_files={hydra_literal(train_files)}",
            f"++data.val_files={hydra_literal(val_files)}",
            f"++actor_rollout_ref.model.path={hydra_literal(model)}",
            f"++critic.model.path={hydra_literal(model)}",
            f"++trainer.n_gpus_per_node={n_gpus_per_node}",
            "++trainer.nnodes=1",
            "++trainer.val_before_train=false",
            f"++trainer.default_local_dir={hydra_literal(str(checkpoint_dir))}",
            f"++trainer.total_epochs={total_epochs}",
            "++trainer.save_freq=1",
            f"++trainer.project_name={project_name}",
            f"++trainer.experiment_name={hydra_literal(run_name)}",
            "++trainer.resume_mode=disable",
        ]
    )

    if not has_any(
        existing_keys,
        "actor_rollout_ref.actor.ppo_micro_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8")
    if not has_any(
        existing_keys,
        "actor_rollout_ref.ref.log_prob_micro_batch_size",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8")
    if not has_any(
        existing_keys,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    ):
        overrides.append("++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8")
    if not has_any(
        existing_keys,
        "critic.forward_micro_batch_size",
        "critic.forward_micro_batch_size_per_gpu",
    ):
        overrides.append("++critic.forward_micro_batch_size_per_gpu=8")
    if not has_any(
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
        overrides.append(f"++trainer.logger={hydra_literal(default_logger)}")

    if not has_any(
        existing_keys,
        "actor_rollout_ref.model.override_config._attn_implementation",
        "actor_rollout_ref.model.override_config.attn_implementation",
    ):
        overrides.append("++actor_rollout_ref.model.override_config.attn_implementation=sdpa")
        overrides.append("++actor_rollout_ref.model.override_config._attn_implementation=sdpa")
    if not has_any(
        existing_keys,
        "critic.model.override_config._attn_implementation",
        "critic.model.override_config.attn_implementation",
    ):
        overrides.append("++critic.model.override_config.attn_implementation=sdpa")
        overrides.append("++critic.model.override_config._attn_implementation=sdpa")

    if extra_override_items:
        overrides.extend(extra_override_items)

    return overrides


def build_train_grpo_command(overrides: list[str]) -> list[str]:
    return ["python", "scripts/train_grpo.py", *overrides]


def build_launcher_env(repo_root: Path, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env) if base_env is not None else os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VERL_LOG_LEVEL", "WARNING")
    env.setdefault("RAY_DEDUP_LOGS", "1")
    env.setdefault("RAY_IGNORE_UNHANDLED_ERRORS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    return env


def pkg_version(pkg_name: str) -> str:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return "not-installed"
