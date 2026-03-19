"""Shared tensor and reward-shape helpers for veRL integration modules."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from verl.workers.config import ActorConfig

from .loss_context import config_get_context as _config_get_context


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


# Public aliases without leading underscore for call sites that prefer clean names.
as_tensor = _as_tensor
infer_sequence_rewards = _infer_sequence_rewards
validate_tensor_shape = _validate_tensor_shape

__all__ = [
    "_as_tensor",
    "_infer_sequence_rewards",
    "_validate_tensor_shape",
    "as_tensor",
    "infer_sequence_rewards",
    "validate_tensor_shape",
]
