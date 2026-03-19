"""
veRL Composable Policy Loss registration: `composer`.

This adapter keeps one policy loss entry in veRL and dispatches clipping,
aggregation, and regularization to grpo_composer core components.
"""

from __future__ import annotations

from typing import Any, Optional

import os
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

from .aggregations_registery import AGG_REGISTRY, VERL_AGG_MODES
from .clip_registery import CLIP_REGISTRY
from .loss_context import (
    config_get as _config_get,
    get_composer_batch_context,
    get_composer_config,
    set_composer_config,
)
from .regularisation_registery import REG_REGISTRY
from .utils import _as_tensor, _infer_sequence_rewards, _validate_tensor_shape







@register_policy_loss("composer")
def compute_composer_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Composer policy loss — dispatches to registered clip / agg / reg components.

    Tensor shapes (all from veRL's FSDP worker):
        old_log_prob:   (B, T)  — log π_θ_old per token, from vLLM rollout
        log_prob:       (B, T)  — log π_θ per token, recomputed in actor forward
        advantages:     (B, T)  — per-token advantage (same A_i broadcast to all tokens of output i)
        response_mask:  (B, T)  — 1 for real response tokens, 0 for padding

    where B = total outputs across all prompts (B = num_prompts × G), flattened.
          T = max response length in this micro-batch (right-padded).

    Shape flow:
        ratio:          (B, T)  — π_θ / π_θ_old per token
        clipped_ratio:  (B, T)  — clip(ratio, 1-ε, 1+ε) per token
        pg_losses:      (B, T)  — max(-A·ρ, -A·clip(ρ)) per token
        pg_loss:        scalar  — aggregated (e.g. token_mean: mean over valid tokens)
        reg_term:       scalar  — regularization penalty
        final loss:     scalar  — pg_loss + β · reg_term
    """
    if config is None:
        raise ValueError("composer policy loss requires actor config")

    if old_log_prob.shape != log_prob.shape:
        raise ValueError(f"old_log_prob/log_prob shape mismatch: {old_log_prob.shape} vs {log_prob.shape}")
    if advantages.shape != log_prob.shape:
        raise ValueError(f"advantages/log_prob shape mismatch: {advantages.shape} vs {log_prob.shape}")
    if response_mask.shape != log_prob.shape:
        raise ValueError(f"response_mask/log_prob shape mismatch: {response_mask.shape} vs {log_prob.shape}")

    composer_dict = kwargs.get("composer_config")
    if not isinstance(composer_dict, dict) or not composer_dict:
        composer_dict = get_composer_config()

    # Fail fast: composer loss must run with explicit composer settings.
    if not isinstance(composer_dict, dict) or not composer_dict:
        raise RuntimeError(
            "Composer loss is active, but worker has no composer config. "
            "Expected trainer to inject batch.meta_info['composer_config'] and "
            "worker update_policy patch to bind it via set_composer_config(...)."
        )

    # Keep the active per-batch config visible to nested clip/agg/reg helpers.
    set_composer_config(composer_dict)

    clip_mode = _config_get(config, "clip_mode", "symmetric", composer_dict)
    agg_mode = _config_get(config, "agg_mode", "token_mean", composer_dict)
    reg_name = _config_get(config, "regularizer", "none", composer_dict)
    reg_coef = float(_config_get(config, "reg_coef", 0.0, composer_dict))
    use_dual_clip = bool(_config_get(config, "use_dual_clip", False, composer_dict))
    clip_ratio_c = float(_config_get(config, "clip_ratio_c", 3.0, composer_dict))

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"🔥 [DEBUG] Loss Compute | Clip: {clip_mode} | Agg: {agg_mode} | Reg: {reg_name}")

    # (B, T) — clamped log-ratio for numerical stability
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    # (B, T) — importance sampling ratio ρ_{i,t} = π_θ / π_θ_old
    ratio = torch.exp(negative_approx_kl)
    # scalar — monitoring metric only, NOT added to loss
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    clip_fn = CLIP_REGISTRY.get(clip_mode)
    if clip_fn is None:
        raise ValueError(f"Unknown clip_mode: {clip_mode}. Options: {list(CLIP_REGISTRY.keys())}")
    # (B, T) — clipped ratio, e.g. symmetric → clamp(ratio, 1-ε, 1+ε)
    clipped_ratio = clip_fn(
        ratio,
        old_log_prob,
        log_prob,
        response_mask,
        config,
        **kwargs,
    )

    # (B, T) — PPO surrogate losses (negative because we maximize objective)
    pg_losses1 = -advantages * ratio             # unclipped: -A_i · ρ_{i,t}
    pg_losses2 = -advantages * clipped_ratio      # clipped:   -A_i · clip(ρ_{i,t})
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # pessimistic bound

    if use_dual_clip:
        pg_losses3 = -advantages * clip_ratio_c
        pg_losses = torch.where(advantages < 0, torch.min(pg_losses3, pg_losses), pg_losses)

    if rollout_is_weights is not None:
        rollout_is_weights = _as_tensor(
            rollout_is_weights, name="rollout_is_weights", device=pg_losses.device
        )
        _validate_tensor_shape(
            rollout_is_weights,
            ndim=(1, 2),
            first_dim=pg_losses.shape[0],
            name="rollout_is_weights",
        )
        if rollout_is_weights.ndim == 1:
            rollout_is_weights = rollout_is_weights.unsqueeze(-1)
        pg_losses = pg_losses * rollout_is_weights

    # (B,) or None — sequence-level reward for agg/reg that need it
    sequence_rewards = _infer_sequence_rewards(response_mask, kwargs, config)

    runtime_context = dict(kwargs)
    runtime_context.update(get_composer_batch_context())
    runtime_context["log_prob"] = log_prob
    if sequence_rewards is not None:
        runtime_context["sequence_rewards"] = sequence_rewards
    runtime_context["composer_config"] = composer_dict

    # scalar — aggregated loss: (B, T) → scalar via token_mean, token_sum, etc.
    if agg_mode in AGG_REGISTRY:
        pg_loss = AGG_REGISTRY[agg_mode](pg_losses, response_mask, config, **runtime_context)
    elif agg_mode in VERL_AGG_MODES:
        pg_loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=response_mask,
            loss_agg_mode=VERL_AGG_MODES[agg_mode],
            **getattr(config, "global_batch_info", {}),
        )
    else:
        raise ValueError(
            f"Unknown agg_mode: {agg_mode}. "
            f"Options: {list(AGG_REGISTRY.keys()) + list(VERL_AGG_MODES.keys())}"
        )

    # scalar — regularization term (e.g. KL divergence)
    reg_term_value = torch.tensor(0.0, device=log_prob.device)
    if reg_name != "none" and reg_coef > 0:
        reg_fn = REG_REGISTRY.get(reg_name)
        if reg_fn is None:
            raise ValueError(f"Unknown regularizer: {reg_name}. Options: {list(REG_REGISTRY.keys())}")
        reg_kwargs = {k: v for k, v in runtime_context.items() if k not in ("log_prob", "old_log_prob", "response_mask", "config")}
        reg_term_value = reg_fn(log_prob, old_log_prob, response_mask, config, **reg_kwargs)
        # scalar — final loss = pg_loss + β · reg_term
        pg_loss = pg_loss + reg_coef * reg_term_value

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    metrics: dict[str, Any] = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_loss": pg_loss.detach().item(),
        "actor/reg_term": reg_term_value.detach().item(),
    }
    if sequence_rewards is not None:
        metrics["actor/sequence_reward_mean"] = sequence_rewards.mean().detach().item()
    return pg_loss, metrics
