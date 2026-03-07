"""Launcher preflight sanity checks.

These checks fail early for known misconfigurations and missing required
signals for certain paper variants.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


_INTEGER_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+)$")
_SCIENTIFIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping, got {type(loaded)} for {path}")
    return loaded


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if _INTEGER_RE.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            return value

    if _FLOAT_RE.fullmatch(value) or _SCIENTIFIC_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            return value

    if value.startswith("[") or value.startswith("{") or value.startswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Support single-quoted literals passed from shell.
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]

    return value


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _deep_copy_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy_dict(v) for v in obj]
    return obj


def parse_overrides_to_dict(overrides: Iterable[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        raw = item.strip()
        if not raw:
            continue
        while raw.startswith("+"):
            raw = raw[1:]
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        _set_nested(parsed, key, _coerce_scalar(value.strip()))
    return parsed


def _merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_copy_dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = _deep_copy_dict(value)
    return merged


def build_effective_config(base_cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    return _merge_dicts(base_cfg, parse_overrides_to_dict(overrides))


def _get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
        return out
    return []


def _read_parquet_columns(path: str) -> set[str] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or candidate.suffix.lower() != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(candidate)
        return set(schema.names)
    except Exception:
        return None


def _collect_required_signal_rules(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    adv_estimator = _get_nested(config, "algorithm.adv_estimator", "grpo")
    pipeline = _normalize_string_list(_get_nested(config, "composer.composer_reward_pipeline", []))

    if adv_estimator == "static_value_grpo":
        rules.append(
            {
                "reason": "adv_estimator=static_value_grpo",
                "any_of": ["reference_rewards", "composer_reference_rewards", "ref_rewards"],
            }
        )

    if adv_estimator == "stratified_grpo":
        rules.append(
            {
                "reason": "adv_estimator=stratified_grpo",
                "any_of": ["strata", "stratum_ids", "trajectory_strata", "composer_strata"],
            }
        )

    if adv_estimator == "decoupled_grpo" or "multi_reward" in pipeline:
        rules.append(
            {
                "reason": "decoupled/multi_reward pipeline",
                "any_of": ["multi_rewards", "composer_multi_rewards", "reward_components"],
            }
        )

    if "posterior" in pipeline:
        rules.append(
            {
                "reason": "posterior reward pipeline",
                "all_of": [
                    "composer_format_rewards",
                    "composer_outcome_rewards",
                    "composer_thinking_rewards",
                ],
            }
        )

    if "rts" in pipeline:
        rules.append(
            {
                "reason": "rts reward pipeline",
                "any_of": ["rts_scores", "composer_rts_scores"],
            }
        )

    return rules


def run_preflight_sanity_checks(
    *,
    config_path: Path,
    overrides: Iterable[str],
    train_file: str,
) -> list[str]:
    """Validate merged config plus dataset signals.

    Returns warnings. Raises ValueError on blocking issues.
    """
    base_cfg = load_yaml_mapping(config_path)
    effective = build_effective_config(base_cfg, overrides)

    errors: list[str] = []
    warnings: list[str] = []

    loss_mode = _get_nested(effective, "actor_rollout_ref.actor.policy_loss.loss_mode")
    legacy_loss_fn = _get_nested(effective, "actor_rollout_ref.actor.loss_fn")
    if loss_mode != "composer":
        if legacy_loss_fn == "composer":
            errors.append(
                "Detected legacy actor.loss_fn=composer without actor.policy_loss.loss_mode=composer. "
                "Use actor.policy_loss.loss_mode for current veRL compatibility."
            )
        else:
            errors.append(
                "actor_rollout_ref.actor.policy_loss.loss_mode must be 'composer' for this repo's training path."
            )

    agg_mode = _get_nested(effective, "composer.agg_mode")
    lambda_learnable = bool(_get_nested(effective, "composer.lambda_learnable", False))
    if agg_mode == "group_learnable" and lambda_learnable:
        errors.append(
            "composer.agg_mode=group_learnable with composer.lambda_learnable=true is not yet supported by "
            "the trainer optimizer path."
        )

    regularizer = _get_nested(effective, "composer.regularizer")
    algorithm_flow = _get_nested(effective, "algorithm.composer_flow")
    composer_flow = _get_nested(effective, "composer.composer_flow")
    if composer_flow and not algorithm_flow:
        warnings.append(
            "composer.composer_flow is set but algorithm.composer_flow is not. Current trainer reads flow from "
            "algorithm.* keys."
        )

    if regularizer == "mutual_info" or algorithm_flow == "info_grpo":
        errors.append(
            "Info-GRPO/mutual_info regularization requires augmented rollout tensors (log_probs_aug/mask_aug). "
            "Default launcher path does not generate these tensors."
        )

    signal_rules = _collect_required_signal_rules(effective)
    if signal_rules:
        columns = _read_parquet_columns(train_file)
        if columns is None:
            errors.append(
                f"Cannot validate required dataset signals for configured variant; readable parquet not found: {train_file}"
            )
        else:
            for rule in signal_rules:
                reason = rule["reason"]
                if "any_of" in rule:
                    options = rule["any_of"]
                    if not any(col in columns for col in options):
                        errors.append(
                            f"Missing required dataset signal for {reason}. Expected any of: {options}."
                        )
                if "all_of" in rule:
                    required = rule["all_of"]
                    missing = [col for col in required if col not in columns]
                    if missing:
                        errors.append(
                            f"Missing required dataset signals for {reason}: {missing}."
                        )

    if errors:
        bullet_lines = "\n".join(f"- {line}" for line in errors)
        raise ValueError(f"Preflight sanity checks failed:\n{bullet_lines}")

    return warnings
