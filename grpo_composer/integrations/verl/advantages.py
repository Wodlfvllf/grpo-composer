"""
veRL Advantage Estimator Registrations.

This adapter registers custom advantage estimators into veRL while handling
shape conversions between veRL batch tensors and grpo_composer core APIs.
"""

from collections import defaultdict
from typing import Any, Callable, Optional

import os

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est

from grpo_composer.core.advantages.decoupled import DecoupledAdvantageFunction
from grpo_composer.core.advantages.difficulty_aware import DifficultyAwareAdvantageFunction
from grpo_composer.core.advantages.kalman import KalmanAdvantageFunction
from grpo_composer.core.advantages.length_corrected import LengthCorrectedAdvantageFunction
from grpo_composer.core.advantages.multi_scale import MultiScaleAdvantageFunction
from grpo_composer.core.advantages.novelty_sharpening import NoveltySharpeningAdvantageFunction
from grpo_composer.core.advantages.standard import StandardAdvantageFunction
from grpo_composer.core.advantages.static_value import StaticValueAdvantageFunction
from grpo_composer.core.advantages.stratified import StratifiedAdvantageFunction
from grpo_composer.core.advantages.unbiased import UnbiasedAdvantageFunction


def _config_get(config: Optional[AlgoConfig], key: str, default):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _validate_inputs(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray) -> None:
    if token_level_rewards.ndim != 2:
        raise ValueError(f"token_level_rewards must be 2D (bs, T), got {token_level_rewards.shape}")
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be 2D (bs, T), got {response_mask.shape}")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            "token_level_rewards and response_mask must share shape, got "
            f"{token_level_rewards.shape} vs {response_mask.shape}"
        )

    if index.ndim != 1:
        raise ValueError(f"index must be 1D (bs,), got {index.shape}")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError(
            f"index length ({len(index)}) must equal batch size ({token_level_rewards.shape[0]})"
        )


def _collect_group_indices(index: np.ndarray, batch_size: int) -> dict[Any, list[int]]:
    group_indices: dict[Any, list[int]] = defaultdict(list)
    for i in range(batch_size):
        group_indices[index[i]].append(i)
    return group_indices


def _sequence_scores(token_level_rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    # Respect response mask to avoid padding tokens contributing to sequence rewards.
    return (token_level_rewards * response_mask).sum(dim=-1)


def _broadcast_sequence_advantages(
    sequence_advantages: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = sequence_advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def _resolve_optional_tensor(kwargs: dict, candidates: tuple[str, ...]) -> Optional[torch.Tensor]:
    for key in candidates:
        value = kwargs.get(key)
        if value is not None:
            return value
    return None


def _as_tensor(value, *, device: torch.device, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor or np.ndarray, got {type(value)}")
    return value.to(device=device, dtype=dtype if dtype is not None else value.dtype)


def _validate_batch_aligned_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    *,
    name: str,
    expected_ndim: tuple[int, ...],
) -> None:
    if tensor.ndim not in expected_ndim:
        raise ValueError(f"{name} must have ndim in {expected_ndim}, got shape {tensor.shape}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{name} batch dimension must be {batch_size}, got {tensor.shape}")


def _compute_groupwise(
    scores: torch.Tensor,
    group_indices: dict[int, list[int]],
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    out = torch.zeros_like(scores)
    for indices in group_indices.values():
        idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
        group_scores = scores[idx].unsqueeze(0)
        group_adv = fn(group_scores)
        if group_adv.ndim != 2 or group_adv.shape[0] != 1 or group_adv.shape[1] != len(indices):
            raise ValueError(
                "Advantage function must return shape (1, group_size), got "
                f"{tuple(group_adv.shape)} for group size {len(indices)}"
            )
        out[idx] = group_adv.squeeze(0)
    return out


# Persistent instance to maintain KRPO filter state across steps.
_kalman_fn: Optional[KalmanAdvantageFunction] = None


@register_adv_est("difficulty_aware_grpo")
def compute_difficulty_aware_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    A = _config_get(config, "difficulty_A", 0.2)
    B_param = _config_get(config, "difficulty_B", 1.0)
    k = _config_get(config, "difficulty_k", 10.0)
    rho_0 = _config_get(config, "difficulty_rho_0", 0.5)

    fn = DifficultyAwareAdvantageFunction(A=A, B=B_param, k=k, rho_0=rho_0, epsilon=epsilon)
    with torch.no_grad():
        sequence_adv = _compute_groupwise(scores, group_indices, fn.compute_advantages)
    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("length_corrected_grpo")
def compute_length_corrected_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    lengths = response_mask.sum(dim=-1)
    group_indices = _collect_group_indices(index, scores.shape[0])

    fn = LengthCorrectedAdvantageFunction(epsilon=epsilon)
    with torch.no_grad():
        sequence_adv = torch.zeros_like(scores)
        for indices in group_indices.values():
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores[idx].unsqueeze(0)
            group_lengths = lengths[idx].unsqueeze(0)
            group_adv = fn.compute_advantages(group_scores, group_lengths)
            if group_adv.ndim != 2 or group_adv.shape != group_scores.shape:
                raise ValueError(
                    "LengthCorrectedAdvantageFunction returned invalid shape "
                    f"{tuple(group_adv.shape)} for group shape {tuple(group_scores.shape)}"
                )
            sequence_adv[idx] = group_adv.squeeze(0)

    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("kalman_grpo")
def compute_kalman_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)

    global _kalman_fn
    Q = _config_get(config, "kalman_Q", 1e-4)
    R = _config_get(config, "kalman_R", 1.0)
    if _kalman_fn is None:
        _kalman_fn = KalmanAdvantageFunction(process_noise=Q, measurement_noise=R, epsilon=epsilon)

    # KRPO baseline is global/stateful; no group padding needed.
    with torch.no_grad():
        sequence_adv = _kalman_fn.compute_advantages(scores.unsqueeze(0)).squeeze(0)
    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("decoupled_grpo")
def compute_decoupled_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    multi_rewards = _resolve_optional_tensor(
        kwargs,
        ("multi_rewards", "composer_multi_rewards", "reward_components"),
    )

    with torch.no_grad():
        if multi_rewards is None:
            raise ValueError(
                "adv_estimator='decoupled_grpo' requires multi-reward inputs. "
                "Provide one of: multi_rewards, composer_multi_rewards, reward_components "
                "with shape (bs, K) or (bs, T, K)."
            )

        multi_rewards = _as_tensor(multi_rewards, device=scores.device, name="multi_rewards")
        _validate_batch_aligned_tensor(
            multi_rewards,
            scores.shape[0],
            name="multi_rewards",
            expected_ndim=(2, 3),
        )
        if multi_rewards.ndim == 3:
            if multi_rewards.shape[:2] != token_level_rewards.shape:
                raise ValueError(
                    "3D multi_rewards must have shape (bs, T, K) aligned with token rewards, got "
                    f"{multi_rewards.shape} vs token shape {token_level_rewards.shape}"
                )
            seq_multi = (multi_rewards * response_mask.unsqueeze(-1)).sum(dim=1)
        else:
            seq_multi = multi_rewards

        fn = DecoupledAdvantageFunction(eps=epsilon)
        sequence_adv = torch.zeros_like(scores)
        for indices in group_indices.values():
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_multi = seq_multi[idx].unsqueeze(0)
            group_adv = fn.compute_advantages(group_multi)
            if group_adv.ndim != 2 or group_adv.shape[0] != 1:
                raise ValueError(
                    f"DecoupledAdvantageFunction returned invalid shape {tuple(group_adv.shape)}"
                )
            if group_adv.shape[1] != len(indices):
                raise ValueError(
                    f"DecoupledAdvantageFunction returned group size {group_adv.shape[1]}, expected {len(indices)}"
                )
            sequence_adv[idx] = group_adv.squeeze(0)

    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("multi_scale_grpo")
def compute_multi_scale_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    tau_min = int(_config_get(config, "ms_tau_min", 2))
    num_scales = int(_config_get(config, "ms_num_scales", 4))
    samples_per_scale = int(_config_get(config, "ms_samples_per_scale", 4))
    weights = _config_get(config, "ms_weights", None)
    
    fn = MultiScaleAdvantageFunction(
        tau_min=tau_min, 
        num_scales=num_scales,
        samples_per_scale=samples_per_scale,
        weights=weights,
        epsilon=epsilon
    )
    with torch.no_grad():
        sequence_adv = _compute_groupwise(scores, group_indices, fn.compute_advantages)
    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("static_value_grpo")
def compute_static_value_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    ref_rewards = _resolve_optional_tensor(
        kwargs,
        ("reference_rewards", "composer_reference_rewards", "ref_rewards"),
    )

    with torch.no_grad():
        if ref_rewards is None:
            raise ValueError(
                "adv_estimator='static_value_grpo' requires reference rewards. "
                "Provide one of: reference_rewards, composer_reference_rewards, ref_rewards "
                "with shape (bs,) or (bs, T)."
            )

        ref_rewards = _as_tensor(ref_rewards, device=scores.device, name="reference_rewards")
        _validate_batch_aligned_tensor(
            ref_rewards,
            scores.shape[0],
            name="reference_rewards",
            expected_ndim=(1, 2),
        )
        if ref_rewards.ndim == 2:
            if ref_rewards.shape != token_level_rewards.shape:
                raise ValueError(
                    "2D reference_rewards must match token reward shape (bs, T), got "
                    f"{ref_rewards.shape} vs {token_level_rewards.shape}"
                )
            seq_ref = (ref_rewards * response_mask).sum(dim=-1)
        else:
            seq_ref = ref_rewards

        fn = StaticValueAdvantageFunction(epsilon=epsilon)
        sequence_adv = torch.zeros_like(scores)
        for indices in group_indices.values():
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores[idx].unsqueeze(0)
            group_ref = seq_ref[idx].unsqueeze(0)
            group_adv = fn.compute_advantages(group_scores, group_ref)
            if group_adv.ndim != 2 or group_adv.shape != group_scores.shape:
                raise ValueError(
                    "StaticValueAdvantageFunction returned invalid shape "
                    f"{tuple(group_adv.shape)} for group shape {tuple(group_scores.shape)}"
                )
            sequence_adv[idx] = group_adv.squeeze(0)

    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("novelty_sharp_grpo")
def compute_novelty_sharpened_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    lam = _config_get(config, "novelty_lambda", 1.0)
    kappa = _config_get(config, "novelty_kappa", 1.0)

    old_log_probs = _resolve_optional_tensor(
        kwargs,
        ("old_log_probs", "old_log_prob", "ref_log_probs", "reference_log_probs"),
    )

    with torch.no_grad():
        if old_log_probs is None:
            fn = StandardAdvantageFunction()
            sequence_adv = _compute_groupwise(scores, group_indices, fn.compute_advantages)
            return _broadcast_sequence_advantages(sequence_adv, response_mask)

        old_log_probs = _as_tensor(old_log_probs, device=scores.device, name="old_log_probs")
        _validate_batch_aligned_tensor(
            old_log_probs,
            scores.shape[0],
            name="old_log_probs",
            expected_ndim=(2,),
        )
        if old_log_probs.shape != token_level_rewards.shape:
            raise ValueError(
                f"old_log_probs must match token shape {token_level_rewards.shape}, got {old_log_probs.shape}"
            )

        fn = NoveltySharpeningAdvantageFunction(lambda_novelty=lam, kappa_clip=kappa, epsilon=epsilon)
        sequence_adv = torch.zeros_like(scores)
        for indices in group_indices.values():
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores[idx].unsqueeze(0)
            group_llk = old_log_probs[idx].unsqueeze(0)
            group_adv = fn.compute_advantages(group_scores, group_llk)
            if group_adv.ndim != 2 or group_adv.shape != group_scores.shape:
                raise ValueError(
                    "NoveltySharpeningAdvantageFunction returned invalid shape "
                    f"{tuple(group_adv.shape)} for group shape {tuple(group_scores.shape)}"
                )
            sequence_adv[idx] = group_adv.squeeze(0)

    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("stratified_grpo")
def compute_stratified_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    alpha = float(_config_get(config, "stratified_alpha", 1.0))
    strata = _resolve_optional_tensor(
        kwargs,
        ("strata", "stratum_ids", "trajectory_strata", "composer_strata"),
    )

    with torch.no_grad():
        if strata is None:
            raise ValueError(
                "adv_estimator='stratified_grpo' requires stratum ids. "
                "Provide one of: strata, stratum_ids, trajectory_strata, composer_strata "
                "with shape (bs,)."
            )

        strata = _as_tensor(strata, device=scores.device, name="strata")
        _validate_batch_aligned_tensor(strata, scores.shape[0], name="strata", expected_ndim=(1,))

        fn = StratifiedAdvantageFunction(alpha=alpha, epsilon=epsilon)
        sequence_adv = torch.zeros_like(scores)
        for indices in group_indices.values():
            idx = torch.tensor(indices, device=scores.device, dtype=torch.long)
            group_scores = scores[idx].unsqueeze(0)
            group_strata = strata[idx].unsqueeze(0)
            group_adv = fn.compute_advantages(group_scores, group_strata)
            if group_adv.ndim != 2 or group_adv.shape != group_scores.shape:
                raise ValueError(
                    f"StratifiedAdvantageFunction returned {tuple(group_adv.shape)} for {tuple(group_scores.shape)}"
                )
            sequence_adv[idx] = group_adv.squeeze(0)

    return _broadcast_sequence_advantages(sequence_adv, response_mask)


@register_adv_est("unbiased_grpo")
def compute_unbiased_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print("🚀 [DEBUG] compute_unbiased_advantage called!")

    _validate_inputs(token_level_rewards, response_mask, index)
    scores = _sequence_scores(token_level_rewards, response_mask)
    group_indices = _collect_group_indices(index, scores.shape[0])

    fn = UnbiasedAdvantageFunction()
    with torch.no_grad():
        sequence_adv = _compute_groupwise(scores, group_indices, fn.compute_advantages) 
    return _broadcast_sequence_advantages(sequence_adv, response_mask)
