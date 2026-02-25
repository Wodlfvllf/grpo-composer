"""
veRL Composable Policy Loss — "composer"

Registers ONE policy loss into veRL that internally dispatches to
configurable sub-components via config keys:

    clip_mode:    "symmetric" | "asymmetric" | "trajectory" | "weighted_trust"
    agg_mode:     "token_mean" | "token_sum" | "global_token" | "weighted_token"
    regularizer:  "none" | "kl" | "weighted_kl"
    reg_coef:     float (regularizer coefficient)

All sub-components import from grpo_composer.core/ — the integration
layer is a thin adapter, not a reimplementation.

Usage in veRL YAML config:
    actor_rollout_ref:
      actor:
        loss_fn: "composer"
        clip_mode: "asymmetric"
        clip_ratio_low: 0.2
        clip_ratio_high: 0.28
        agg_mode: "global_token"
        regularizer: "kl"
        reg_coef: 0.01
"""

from typing import Any, Optional

import torch

from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss
from verl.workers.config import ActorConfig
import verl.utils.torch_functional as verl_F

# ── Import YOUR core implementations ──
from grpo_composer.core.clipping.symmetric import SymmetricClippingMechanism
from grpo_composer.core.clipping.asymmetric import AsymmetricClippingMechanism
from grpo_composer.core.clipping.trajectory_level import TrajectoryLevelClippingMechanism
from grpo_composer.core.clipping.weighted_trust import WeightedTrustRegionClippingMechanism

from grpo_composer.core.aggregation.token_mean import TokenMeanAggregation
from grpo_composer.core.aggregation.token_sum import TokenSumAggregation
from grpo_composer.core.aggregation.global_token import GlobalTokenAggregation
from grpo_composer.core.aggregation.weighted_token import WeightedTokenAggregation

from grpo_composer.core.regularizers.kl_divergence import (
    KLDivergenceRegularizer,
    WeightedKLDivergenceRegularizer,
)
from grpo_composer.core.regularizers.preference import PreferenceRegularizer
from grpo_composer.core.regularizers.mutual_information import MutualInformationRegularizer


# ═══════════════════════════════════════════════════════════════
# CLIPPING SUB-REGISTRY
#
# Each adapter wraps a core/clipping/ class, translating
# veRL's (bs, response_length) tensors to the class API.
# ═══════════════════════════════════════════════════════════════

def clip_symmetric(ratio, old_log_prob, log_prob, response_mask, config):
    """Wraps core/clipping/symmetric.py — clip(ρ, 1-ε, 1+ε)"""
    eps = config.clip_ratio
    clipper = SymmetricClippingMechanism(epsilon=eps)
    return clipper.clip(ratio)


def clip_asymmetric(ratio, old_log_prob, log_prob, response_mask, config):
    """Wraps core/clipping/asymmetric.py — clip(ρ, 1-ε_l, 1+ε_h)"""
    eps_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    eps_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clipper = AsymmetricClippingMechanism(epsilon_lower=eps_low, epsilon_upper=eps_high)
    return clipper.clip(ratio)


def clip_trajectory(ratio, old_log_prob, log_prob, response_mask, config):
    """
    Wraps core/clipping/trajectory_level.py
    Computes trajectory-level ratio from log-probs, then upper-only clips.
    Returns (bs, 1) for broadcasting across tokens.
    """
    clip_upper = config.clip_ratio_high if config.clip_ratio_high is not None else 1 + config.clip_ratio
    clipper = TrajectoryLevelClippingMechanism(clip_upper=clip_upper)
    # Use the clipper's trajectory ratio computation from log-probs
    clipped = clipper.clip(log_probs=log_prob, ref_log_probs=old_log_prob, attention_mask=response_mask)
    # Reshape to (bs, 1) for broadcasting in PPO surrogate
    if clipped.dim() == 1:
        clipped = clipped.unsqueeze(-1)
    return clipped


def clip_weighted_trust(ratio, old_log_prob, log_prob, response_mask, config):
    """
    Wraps core/clipping/weighted_trust.py
    Per-token confidence-weighted dynamic clipping bounds.
    """
    alpha = config.get("tr_alpha", 1.0)
    tau = config.get("tr_tau", 1.0)
    mu = config.get("tr_mu", 0.5)
    base_eps = config.clip_ratio
    clipper = WeightedTrustRegionClippingMechanism(
        alpha=alpha, tau=tau, mu=mu, clip_epsilon=base_eps
    )
    token_probs = torch.exp(log_prob)
    return clipper.clip_with_dynamic_bounds(ratio, token_probs)


CLIP_REGISTRY = {
    "symmetric": clip_symmetric,
    "asymmetric": clip_asymmetric,
    "trajectory": clip_trajectory,
    "weighted_trust": clip_weighted_trust,
}


# ═══════════════════════════════════════════════════════════════
# AGGREGATION SUB-REGISTRY
#
# Each adapter wraps a core/aggregation/ class, calling its
# .aggregate(loss_per_token, mask) method.
# ═══════════════════════════════════════════════════════════════

def agg_token_mean(loss_mat, mask, config):
    """Wraps core/aggregation/token_mean.py"""
    return TokenMeanAggregation().aggregate(loss_mat, mask)


def agg_token_sum(loss_mat, mask, config):
    """Wraps core/aggregation/token_sum.py"""
    return TokenSumAggregation().aggregate(loss_mat, mask)


def agg_global_token(loss_mat, mask, config):
    """Wraps core/aggregation/global_token.py"""
    return GlobalTokenAggregation().aggregate(loss_mat, mask)


def agg_weighted_token(loss_mat, mask, config):
    """Wraps core/aggregation/weighted_token.py — needs log_probs from config"""
    log_probs = config.get("_composer_log_prob", None)
    agg = WeightedTokenAggregation(
        alpha=config.get("tr_alpha", 1.0),
        tau=config.get("tr_tau", 1.0),
        mu=config.get("tr_mu", 0.5),
    )
    return agg.aggregate(loss_mat, mask, log_probs=log_probs)


AGG_REGISTRY = {
    "token_mean": agg_token_mean,
    "token_sum": agg_token_sum,
    "global_token": agg_global_token,
    "weighted_token": agg_weighted_token,
}

# veRL native modes that can be used via agg_loss() directly
VERL_AGG_MODES = {
    "verl_token_mean": "token-mean",
    "verl_seq_mean_token_sum": "seq-mean-token-sum",
    "verl_seq_mean_token_mean": "seq-mean-token-mean",
}


# ═══════════════════════════════════════════════════════════════
# REGULARIZER SUB-REGISTRY
#
# Each adapter wraps a core/regularizers/ class.
# Note: veRL's tensors are (bs, response_length), while
# core/ regularizers expect (B, G, T). For KL, the math is
# identical on flat tensors since it's per-token.
# ═══════════════════════════════════════════════════════════════

def reg_none(log_prob, old_log_prob, response_mask, config):
    """No regularization."""
    return torch.tensor(0.0, device=log_prob.device)


def reg_kl(log_prob, old_log_prob, response_mask, config):
    """Wraps core/regularizers/kl_divergence.py — KLDivergenceRegularizer"""
    regularizer = KLDivergenceRegularizer()
    # Core expects (B, G, T) but KL is per-token so (bs, T) works identically
    # We pass dummy rewards since KL doesn't use them
    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        mask=response_mask,
        rewards=dummy_rewards,
    )


def reg_weighted_kl(log_prob, old_log_prob, response_mask, config):
    """Wraps core/regularizers/kl_divergence.py — WeightedKLDivergenceRegularizer"""
    regularizer = WeightedKLDivergenceRegularizer()
    # Compute confidence weights for TR-GRPO style weighted KL
    alpha = config.get("tr_alpha", 1.0)
    tau = config.get("tr_tau", 1.0)
    mu = config.get("tr_mu", 0.5)
    token_probs = torch.exp(log_prob)
    weights = torch.clamp(alpha * (torch.sigmoid(token_probs / tau) - mu), 0.5, 1.5)

    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        mask=response_mask,
        rewards=dummy_rewards,
        weights=weights,
    )


def reg_preference(log_prob, old_log_prob, response_mask, config):
    """
    Wraps core/regularizers/preference.py — PreferenceRegularizer (AMIR-GRPO)

    Requires rewards to be injected into config via _composer_rewards.
    Constructs implicit preference pairs from reward rankings and applies
    DPO-style contrastive loss.
    """
    beta_dpo = config.get("beta_dpo", 0.1)
    delta_reward = config.get("delta_reward", 0.0)
    regularizer = PreferenceRegularizer(beta_dpo=beta_dpo, delta_reward=delta_reward)

    # Rewards must be injected by the caller (compute_composer_loss)
    rewards = config.get("_composer_rewards", torch.zeros(log_prob.shape[0], device=log_prob.device))

    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        rewards=rewards,
        mask=response_mask,
    )


def reg_mutual_info(log_prob, old_log_prob, response_mask, config):
    """
    Wraps core/regularizers/mutual_information.py — MutualInformationRegularizer (Info-GRPO)

    Requires augmented trajectory data injected via config:
        config._composer_log_probs_aug: (bs, T) log-probs from z-conditioned generation
        config._composer_mask_aug: (bs, T) mask for augmented generation

    These must be provided by the training loop which handles dual-sampling
    (T_ori from π_θ and T_aug from π_θ(·|z)).
    """
    alpha = config.get("info_alpha", 1.0)
    regularizer = MutualInformationRegularizer(alpha=alpha)

    log_probs_aug = config.get("_composer_log_probs_aug", None)
    mask_aug = config.get("_composer_mask_aug", None)

    if log_probs_aug is None or mask_aug is None:
        # Fallback: no augmented data, return zero (graceful degradation)
        return torch.tensor(0.0, device=log_prob.device)

    # Dummy rewards (MI regularizer doesn't use them directly)
    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)

    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        rewards=dummy_rewards,
        mask=response_mask,
        log_probs_aug=log_probs_aug,
        mask_aug=mask_aug,
    )


REG_REGISTRY = {
    "none": reg_none,
    "kl": reg_kl,
    "weighted_kl": reg_weighted_kl,
    "preference": reg_preference,
    "mutual_info": reg_mutual_info,
}


# ═══════════════════════════════════════════════════════════════
# THE COMPOSER LOSS
# ═══════════════════════════════════════════════════════════════

@register_policy_loss("composer")
def compute_composer_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Composable GRPO loss. Selects clipping, aggregation, and regularizer
    from internal sub-registries based on config keys.

    All sub-components delegate to grpo_composer.core/ implementations.
    This function only orchestrates the pipeline:
        ratio → clip (from core/clipping/) → PPO surrogate
              → aggregate (from core/aggregation/) → + regularizer (from core/regularizers/)

    Config keys:
        clip_mode:    "symmetric" | "asymmetric" | "trajectory" | "weighted_trust"
        agg_mode:     "token_mean" | "token_sum" | "global_token" | "weighted_token"
                      or "verl_token_mean" | "verl_seq_mean_token_sum" | "verl_seq_mean_token_mean"
        regularizer:  "none" | "kl" | "weighted_kl"
        reg_coef:     regularizer coefficient (float)
        use_dual_clip: enable dual-clip PPO (bool)
        clip_ratio_c:  dual-clip lower bound (float)
    """
    assert config is not None

    # ── Read config ──
    clip_mode = config.get("clip_mode", "symmetric")
    agg_mode = config.get("agg_mode", "token_mean")
    reg_name = config.get("regularizer", "none")
    reg_coef = config.get("reg_coef", 0.0)
    use_dual_clip = config.get("use_dual_clip", False)
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    # ── Step 1: Ratio ──
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ── Step 2: Clipping (dispatches to core/clipping/) ──
    clip_fn = CLIP_REGISTRY.get(clip_mode)
    if clip_fn is None:
        raise ValueError(f"Unknown clip_mode: {clip_mode}. Options: {list(CLIP_REGISTRY.keys())}")
    clipped_ratio = clip_fn(ratio, old_log_prob, log_prob, response_mask, config)

    # ── Step 3: PPO surrogate ──
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Dual-clip: lower bound for negative advantages
    if use_dual_clip:
        pg_losses3 = -advantages * clip_ratio_c
        pg_losses = torch.where(advantages < 0, torch.min(pg_losses3, pg_losses), pg_losses)

    # Rollout importance sampling correction
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # ── Step 4: Aggregation (dispatches to core/aggregation/) ──
    if agg_mode in AGG_REGISTRY:
        pg_loss = AGG_REGISTRY[agg_mode](pg_losses, response_mask, config)
    elif agg_mode in VERL_AGG_MODES:
        pg_loss = agg_loss(
            loss_mat=pg_losses, loss_mask=response_mask,
            loss_agg_mode=VERL_AGG_MODES[agg_mode],
            **config.global_batch_info
        )
    else:
        raise ValueError(
            f"Unknown agg_mode: {agg_mode}. "
            f"Options: {list(AGG_REGISTRY.keys()) + list(VERL_AGG_MODES.keys())}"
        )

    # ── Step 5: Regularizer (dispatches to core/regularizers/) ──
    if reg_name != "none" and reg_coef > 0:
        reg_fn = REG_REGISTRY.get(reg_name)
        if reg_fn is None:
            raise ValueError(f"Unknown regularizer: {reg_name}. Options: {list(REG_REGISTRY.keys())}")
        reg_term = reg_fn(log_prob, old_log_prob, response_mask, config)
        pg_loss = pg_loss + reg_coef * reg_term

    # ── Metrics ──
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/clip_mode": clip_mode,
        "actor/agg_mode": agg_mode,
        "actor/regularizer": reg_name,
    }
    return pg_loss, metrics
