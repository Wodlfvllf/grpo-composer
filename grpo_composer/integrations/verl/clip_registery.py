"""Composer clipping mode registry."""

from __future__ import annotations

import torch

from grpo_composer.core.clipping.asymmetric import AsymmetricClippingMechanism
from grpo_composer.core.clipping.symmetric import SymmetricClippingMechanism
from grpo_composer.core.clipping.trajectory_level import TrajectoryLevelClippingMechanism
from grpo_composer.core.clipping.weighted_trust import WeightedTrustRegionClippingMechanism

from .loss_context import config_get as _config_get


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
    alpha = _config_get(config, "tr_alpha", 1.0)
    tau = _config_get(config, "tr_tau", 1.0)
    mu = _config_get(config, "tr_mu", 0.5)
    base_eps = _config_get(config, "clip_ratio", 0.2)
    clipper = WeightedTrustRegionClippingMechanism(alpha=alpha, tau=tau, mu=mu, clip_epsilon=base_eps)
    token_probs = torch.exp(log_prob)
    return clipper.clip_with_dynamic_bounds(ratio, token_probs)


CLIP_REGISTRY = {
    "symmetric": clip_symmetric,
    "asymmetric": clip_asymmetric,
    "trajectory": clip_trajectory,
    "weighted_trust": clip_weighted_trust,
}
