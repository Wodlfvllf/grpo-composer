"""Composer clipping mode registry."""

from __future__ import annotations

import torch

from grpo_composer.core.clipping.asymmetric import AsymmetricClippingMechanism
from grpo_composer.core.clipping.symmetric import SymmetricClippingMechanism
from grpo_composer.core.clipping.trajectory_level import TrajectoryLevelClippingMechanism
from grpo_composer.core.clipping.weighted_trust import WeightedTrustRegionClippingMechanism

from .loss_context import config_get as _config_get
from .utils import (
    _as_tensor,
    _compute_tr_token_weights,
    _resolve_tr_weight_bounds,
    _validate_tensor_shape,
)


# ---------------------------------------------------------------------------
# Clipping registry
# ---------------------------------------------------------------------------


def clip_symmetric(ratio, old_log_prob, log_prob, response_mask, config, **kwargs):
    eps = _config_get(config, "clip_ratio", 0.2)
    clipper = SymmetricClippingMechanism(epsilon=eps)
    return clipper.clip(ratio)


def clip_asymmetric(ratio, old_log_prob, log_prob, response_mask, config, **kwargs):
    eps = _config_get(config, "clip_ratio", 0.2)
    eps_low = _config_get(config, "clip_ratio_low", eps)
    eps_high = _config_get(config, "clip_ratio_high", eps)
    clipper = AsymmetricClippingMechanism(epsilon_lower=eps_low, epsilon_upper=eps_high)
    return clipper.clip(ratio)


def clip_trajectory(ratio, old_log_prob, log_prob, response_mask, config, **kwargs):
    clip_ratio = _config_get(config, "clip_ratio", 0.2)
    clip_upper = _config_get(config, "clip_ratio_high", 1 + clip_ratio)
    clipper = TrajectoryLevelClippingMechanism(clip_upper=clip_upper)
    clipped = clipper.clip(log_probs=log_prob, ref_log_probs=old_log_prob, attention_mask=response_mask)
    if clipped.ndim == 1:
        clipped = clipped.unsqueeze(-1)
    return clipped


def clip_weighted_trust(ratio, old_log_prob, log_prob, response_mask, config, **kwargs):
    alpha = _config_get(config, "tr_alpha", 2.0)
    tau = _config_get(config, "tr_tau", 9.0)
    mu = _config_get(config, "tr_mu", 0.25)
    base_eps = _config_get(config, "clip_ratio", 0.2)
    eps_low = _config_get(config, "clip_ratio_low", base_eps)
    eps_high = _config_get(config, "clip_ratio_high", base_eps)
    weight_lower, weight_upper = _resolve_tr_weight_bounds(config)

    token_weights = kwargs.get("tr_token_weights")
    if token_weights is None:
        token_weights = kwargs.get("token_weights")
    if token_weights is not None:
        token_weights = _as_tensor(token_weights, name="tr_token_weights", device=ratio.device)
        _validate_tensor_shape(
            token_weights, ndim=(2,), first_dim=ratio.shape[0], name="tr_token_weights"
        )
        if token_weights.shape != ratio.shape:
            raise ValueError(
                f"tr_token_weights shape must match ratio shape, got {token_weights.shape} vs {ratio.shape}"
            )
    else:
        token_weights = _compute_tr_token_weights(log_prob, config)

    clipper = WeightedTrustRegionClippingMechanism(
        alpha=alpha,
        tau=tau,
        mu=mu,
        weight_lower=weight_lower,
        weight_upper=weight_upper,
        clip_epsilon=base_eps,
        clip_epsilon_lower=eps_low,
        clip_epsilon_upper=eps_high,
    )
    token_probs = torch.exp(log_prob)
    return clipper.clip(ratio, token_probs=token_probs, weights=token_weights)


CLIP_REGISTRY = {
    "symmetric": clip_symmetric,
    "asymmetric": clip_asymmetric,
    "trajectory": clip_trajectory,
    "weighted_trust": clip_weighted_trust,
}
