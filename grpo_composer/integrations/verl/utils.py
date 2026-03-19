"""Shared tensor and reward-shape helpers for veRL integration modules."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from verl.workers.config import ActorConfig

from .loss_context import (
    config_get as _config_get,
    config_get_context as _config_get_context,
)


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
    "_compute_tr_token_weights",
    "_infer_sequence_rewards",
    "_resolve_tr_weight_bounds",
    "_validate_tensor_shape",
    "as_tensor",
    "compute_tr_token_weights",
    "infer_sequence_rewards",
    "resolve_tr_weight_bounds",
    "validate_tensor_shape",
]
