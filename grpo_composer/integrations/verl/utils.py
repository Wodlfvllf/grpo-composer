"""Shared tensor and reward-shape helpers for veRL integration modules."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from verl.workers.config import ActorConfig

from .loss_context import (
    config_get as _config_get,
    config_get_context as _config_get_context,
)


# ---------------------------------------------------------------------------
# Config / debug helpers (consolidated from trainer.py, composer_dp_actor.py,
# and rewards_registery.py).
# ---------------------------------------------------------------------------


def strict_validation_enabled() -> bool:
    return os.environ.get("GRPO_COMPOSER_STRICT_VALIDATION", "1") != "0"


def shape_debug(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"torch{tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"np{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def cfg_get(config: Any, key: str, default=None):
    """Look up ``key`` on ``config`` with fallback to global composer config."""
    from .loss_context import get_composer_config

    val = None
    if config is not None:
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                val = getter(key, None)
            except TypeError:
                pass
        if val is None:
            val = getattr(config, key, None)

    if val is not None:
        return val

    composer_cfg = get_composer_config()
    if key in composer_cfg and composer_cfg[key] is not None:
        return composer_cfg[key]

    return default


def cfg_get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for part in path:
        if current is None:
            return default
        current = cfg_get(current, part, None)
    return default if current is None else current


def to_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def extract_rollout_n(meta_info: Any) -> Optional[int]:
    value = None
    if isinstance(meta_info, dict):
        value = meta_info.get("rollout_n", None)
        if value is None:
            value = meta_info.get("n", None)
    else:
        getter = getattr(meta_info, "get", None)
        if callable(getter):
            try:
                value = getter("rollout_n", None)
            except Exception:
                value = None
            if value is None:
                try:
                    value = getter("n", None)
                except Exception:
                    value = None

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


# Underscore aliases for legacy call sites.
_strict_validation_enabled = strict_validation_enabled
_shape_debug = shape_debug
_cfg_get = cfg_get
_cfg_get_nested = cfg_get_nested
_to_bool_flag = to_bool_flag
_extract_rollout_n = extract_rollout_n


def _validate_tensor_shape(
    tensor: torch.Tensor,
    *,
    ndim: tuple[int, ...],
    first_dim: int,
    name: str,
) -> None:
    if tensor.ndim not in ndim:
        raise ValueError(f"{name} must have ndim in {ndim}, got {tensor.shape}")
    if tensor.shape[0] != first_dim:
        raise ValueError(
            f"{name} first dimension must match batch size ({first_dim}), got {tensor.shape}"
        )


def _as_tensor(value: Any, *, name: str, device: torch.device) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor or np.ndarray, got {type(value)}")
    return value.to(device=device)


def _infer_sequence_rewards(
    response_mask: torch.Tensor,
    kwargs: dict[str, Any],
    config: Optional[ActorConfig],
) -> Optional[torch.Tensor]:
    sequence_rewards = kwargs.get("sequence_rewards")
    if sequence_rewards is None:
        sequence_rewards = kwargs.get("composer_sequence_rewards")
    if sequence_rewards is not None:
        sequence_rewards = _as_tensor(sequence_rewards, name="sequence_rewards", device=response_mask.device)
        _validate_tensor_shape(
            sequence_rewards,
            ndim=(1,),
            first_dim=response_mask.shape[0],
            name="sequence_rewards",
        )
        return sequence_rewards

    token_level_rewards = kwargs.get("token_level_rewards")
    if token_level_rewards is None:
        token_level_rewards = kwargs.get("composer_token_level_rewards")
    if token_level_rewards is not None:
        token_level_rewards = _as_tensor(
            token_level_rewards, name="token_level_rewards", device=response_mask.device
        )
        _validate_tensor_shape(
            token_level_rewards,
            ndim=(2,),
            first_dim=response_mask.shape[0],
            name="token_level_rewards",
        )
        if token_level_rewards.shape != response_mask.shape:
            raise ValueError(
                "token_level_rewards must match response_mask shape, got "
                f"{token_level_rewards.shape} vs {response_mask.shape}"
            )
        return (token_level_rewards * response_mask).sum(dim=-1)

    config_rewards = _config_get_context(config, "_composer_rewards", None)
    if config_rewards is not None:
        config_rewards = _as_tensor(config_rewards, name="_composer_rewards", device=response_mask.device)
        _validate_tensor_shape(
            config_rewards,
            ndim=(1,),
            first_dim=response_mask.shape[0],
            name="_composer_rewards",
        )
        return config_rewards

    return None


def _resolve_tr_weight_bounds(
    config: Optional[ActorConfig],
    *,
    composer_dict: Optional[dict[str, Any]] = None,
) -> tuple[float, float]:
    lower = _config_get(config, "tr_l", None, composer_dict)
    if lower is None:
        lower = _config_get(config, "tr_L", None, composer_dict)
    if lower is None:
        lower = _config_get(config, "tr_weight_lower", 1.0, composer_dict)

    upper = _config_get(config, "tr_u", None, composer_dict)
    if upper is None:
        upper = _config_get(config, "tr_U", None, composer_dict)
    if upper is None:
        upper = _config_get(config, "tr_weight_upper", 1.4, composer_dict)

    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        raise ValueError(f"TR weight bounds invalid: lower={lower} > upper={upper}")
    return lower, upper


def _compute_tr_token_weights(
    log_prob: torch.Tensor,
    config: Optional[ActorConfig],
    *,
    composer_dict: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    alpha = float(_config_get(config, "tr_alpha", 2.0, composer_dict))
    tau = float(_config_get(config, "tr_tau", 9.0, composer_dict))
    mu = float(_config_get(config, "tr_mu", 0.25, composer_dict))
    if tau <= 0:
        raise ValueError(f"tr_tau must be > 0, got {tau}")
    lower, upper = _resolve_tr_weight_bounds(config, composer_dict=composer_dict)

    # Stop-gradient: TR-GRPO weights should not be optimized through log-probs.
    token_probs = torch.exp(log_prob.detach())
    raw = alpha * (torch.sigmoid(token_probs / tau) - mu)
    return torch.clamp(raw, min=lower, max=upper)


# Public aliases without leading underscore for call sites that prefer clean names.
as_tensor = _as_tensor
infer_sequence_rewards = _infer_sequence_rewards
validate_tensor_shape = _validate_tensor_shape
resolve_tr_weight_bounds = _resolve_tr_weight_bounds
compute_tr_token_weights = _compute_tr_token_weights

__all__ = [
    "_as_tensor",
    "_cfg_get",
    "_cfg_get_nested",
    "_compute_tr_token_weights",
    "_extract_rollout_n",
    "_infer_sequence_rewards",
    "_resolve_tr_weight_bounds",
    "_shape_debug",
    "_strict_validation_enabled",
    "_to_bool_flag",
    "_validate_tensor_shape",
    "as_tensor",
    "cfg_get",
    "cfg_get_nested",
    "compute_tr_token_weights",
    "extract_rollout_n",
    "infer_sequence_rewards",
    "resolve_tr_weight_bounds",
    "shape_debug",
    "strict_validation_enabled",
    "to_bool_flag",
    "validate_tensor_shape",
]
