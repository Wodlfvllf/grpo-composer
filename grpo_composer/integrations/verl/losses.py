"""
veRL Policy Loss Registrations

Registers grpo_composer's custom policy losses into veRL's
POLICY_LOSS_REGISTRY so they can be selected via config:
    actor_rollout_ref.actor.loss_fn: "dapo_loss"

All functions follow veRL's expected signature:
    (old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_is_weights)
    → (loss, metrics_dict)
"""

from typing import Any, Optional

import torch

from verl.trainer.ppo.core_algos import register_policy_loss, agg_loss
from verl.workers.config import ActorConfig
import verl.utils.torch_functional as verl_F


# ──────────────────────────────────────────────────────────────
# 1. DAPO Loss (Asymmetric Clipping)
# ──────────────────────────────────────────────────────────────
@register_policy_loss("dapo_loss")
def compute_dapo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    DAPO: Asymmetric clipping with different ε_low and ε_high.
    Higher upper bound allows more aggressive probability increases,
    preventing entropy collapse.

    Config keys (via ActorConfig):
        clip_ratio_low:  lower ε (default 0.2)
        clip_ratio_high: upper ε (default 0.28)
    """
    assert config is not None
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else 0.2
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else 0.28

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # Asymmetric clipping
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, metrics


# ──────────────────────────────────────────────────────────────
# 2. Trajectory-Level Clipping Loss (TIC-GRPO)
# ──────────────────────────────────────────────────────────────
@register_policy_loss("tic_grpo_loss")
def compute_tic_grpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    TIC-GRPO: Trajectory-level ratio with upper-only clipping.
    Uses product of token ratios (trajectory probability ratio)
    instead of per-token clipping.

    Config keys:
        clip_ratio_high: upper bound for trajectory ratio (default 1.28)
    """
    assert config is not None
    clip_upper = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Trajectory-level log ratio
    token_log_ratio = log_prob - old_log_prob
    token_log_ratio = torch.clamp(token_log_ratio, min=-20.0, max=20.0)

    # Sum over tokens → trajectory log ratio, then exp
    traj_log_ratio = (token_log_ratio * response_mask).sum(dim=-1, keepdim=True)  # (bs, 1)
    traj_ratio = torch.exp(traj_log_ratio)  # (bs, 1)

    # Upper-only clipping
    clipped_ratio = torch.clamp(traj_ratio, max=clip_upper)

    # Broadcast: advantages is (bs, response_length), traj_ratio is (bs, 1)
    pg_losses1 = -advantages * traj_ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    ppo_kl = verl_F.masked_mean(
        -(log_prob - old_log_prob), response_mask
    )
    pg_clipfrac = verl_F.masked_mean(
        (traj_ratio > clip_upper).float().expand_as(response_mask), response_mask
    )

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, metrics


# ──────────────────────────────────────────────────────────────
# 3. Weighted Trust Region Loss (TR-GRPO)
# ──────────────────────────────────────────────────────────────
@register_policy_loss("tr_grpo_loss")
def compute_tr_grpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    TR-GRPO: Per-token confidence-weighted trust region.
    Dynamic clipping bounds based on token probability.

    Config keys:
        tr_alpha: weight scaling (default 1.0)
        tr_tau: sigmoid temperature (default 1.0)
        tr_mu: sigmoid offset (default 0.5)
        clip_ratio: base clip epsilon (default 0.2)
    """
    assert config is not None
    alpha = config.get("tr_alpha", 1.0)
    tau = config.get("tr_tau", 1.0)
    mu = config.get("tr_mu", 0.5)
    clip_eps = config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # Token-level confidence weights
    token_probs = torch.exp(log_prob)
    scaled = token_probs / tau
    weights = torch.clamp(alpha * (torch.sigmoid(scaled) - mu), 0.5, 1.5)

    # Dynamic bounds: confident tokens → tighter, uncertain → looser
    epsilon_lower = clip_eps / weights
    epsilon_upper = clip_eps * weights

    lower_bound = 1 - epsilon_lower
    upper_bound = 1 + epsilon_upper

    clipped_ratio = torch.max(torch.min(ratio, upper_bound), lower_bound)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac = verl_F.masked_mean(
        ((ratio > upper_bound) | (ratio < lower_bound)).float(), response_mask
    )

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
    return pg_loss, metrics


# ──────────────────────────────────────────────────────────────
# 4. Unified Composer Loss (configurable clipping + KL)
# ──────────────────────────────────────────────────────────────
@register_policy_loss("composer_unified")
def compute_composer_unified_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Unified grpo_composer loss. Combines:
    - Configurable clip mode: "symmetric", "asymmetric", "dual_clip"
    - Optional token-level KL penalty (as separate regularizer)

    Config keys:
        clip_mode:        "symmetric" | "asymmetric" | "dual_clip" (default "symmetric")
        clip_ratio:       base ε
        clip_ratio_low:   ε_low for asymmetric
        clip_ratio_high:  ε_high for asymmetric
        clip_ratio_c:     lower bound for dual_clip (default 3.0)
        use_kl_loss:      whether to add explicit KL (default False)
        kl_loss_coef:     KL coefficient (default 0.1)
    """
    assert config is not None
    clip_mode = config.get("clip_mode", "symmetric")
    clip_ratio = config.clip_ratio
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Dual-clip: lower bound for negative advantages
    if clip_mode == "dual_clip":
        clip_c = config.get("clip_ratio_c", 3.0)
        pg_losses3 = -advantages * clip_c
        pg_losses = torch.where(advantages < 0, torch.min(pg_losses3, pg_losses), pg_losses)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # Optional explicit KL regularizer
    use_kl = config.get("use_kl_loss", False)
    if use_kl:
        kl_coef = config.get("kl_loss_coef", 0.1)
        pg_loss = pg_loss + kl_coef * ppo_kl

    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/clip_mode": clip_mode,
    }
    return pg_loss, metrics
