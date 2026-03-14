"""
veRL Composable Policy Loss registration: `composer`.

This adapter keeps one policy loss entry in veRL and dispatches clipping,
aggregation, and regularization to grpo_composer core components.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import os
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

from grpo_composer.core.aggregation.difficulty_weighted import DifficultyWeightedAggregation
from grpo_composer.core.aggregation.global_token import GlobalTokenAggregation
from grpo_composer.core.aggregation.group_learnable import GroupLearnableAggregation
from grpo_composer.core.aggregation.group_uniform import GroupUniformAggregation
from grpo_composer.core.aggregation.token_mean import TokenMeanAggregation
from grpo_composer.core.aggregation.token_sum import TokenSumAggregation
from grpo_composer.core.aggregation.trajectory_level import TrajectoryLevelAggregation
from grpo_composer.core.aggregation.weighted_token import WeightedTokenAggregation
from grpo_composer.core.clipping.asymmetric import AsymmetricClippingMechanism
from grpo_composer.core.clipping.symmetric import SymmetricClippingMechanism
from grpo_composer.core.clipping.trajectory_level import TrajectoryLevelClippingMechanism
from grpo_composer.core.clipping.weighted_trust import WeightedTrustRegionClippingMechanism
from grpo_composer.core.regularizers.kl_divergence import (
    KLDivergenceRegularizer,
    WeightedKLDivergenceRegularizer,
)
from grpo_composer.core.regularizers.log_weight import LogWeightRegularizer
from grpo_composer.core.regularizers.mutual_information import MutualInformationRegularizer
from grpo_composer.core.regularizers.preference import PreferenceRegularizer


# Module-level composer config: stores keys like clip_mode, agg_mode, regularizer
# that can't live in FSDPActorConfig (strict dataclass rejects unknown fields).
# Set by ComposerRayPPOTrainer._inject_composer_config() in the TaskRunner,
# or lazily resolved from Hydra's global config in FSDP worker processes.
_COMPOSER_CONFIG: dict[str, Any] = {}
_COMPOSER_CONFIG_LOADED: bool = False
_COMPOSER_BATCH_CONTEXT: dict[str, Any] = {}


def set_composer_config(config_dict: dict[str, Any]) -> None:
    """Store composer config globally in the current worker process."""
    global _COMPOSER_CONFIG, _COMPOSER_CONFIG_LOADED
    _COMPOSER_CONFIG = dict(config_dict or {})
    _COMPOSER_CONFIG_LOADED = True


def _ensure_composer_config_loaded() -> None:
    """Best-effort env fallback for worker processes that inherit runtime env."""
    global _COMPOSER_CONFIG, _COMPOSER_CONFIG_LOADED
    if _COMPOSER_CONFIG_LOADED:
        return

    _COMPOSER_CONFIG_LOADED = True
    raw = os.environ.get("GRPO_COMPOSER_CONFIG")
    if not raw:
        return
    try:
        import json

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            _COMPOSER_CONFIG = parsed
    except Exception:
        pass


def get_composer_config() -> dict[str, Any]:
    _ensure_composer_config_loaded()
    return dict(_COMPOSER_CONFIG)


def set_composer_batch_context(context: dict[str, Any]) -> None:
    """Store per-update worker context (uid/correctness/etc.) for loss dispatch."""
    global _COMPOSER_BATCH_CONTEXT
    _COMPOSER_BATCH_CONTEXT = dict(context or {})


def get_composer_batch_context() -> dict[str, Any]:
    return dict(_COMPOSER_BATCH_CONTEXT)


def clear_composer_batch_context() -> None:
    global _COMPOSER_BATCH_CONTEXT
    _COMPOSER_BATCH_CONTEXT = {}


def _config_get(
    config: Optional[ActorConfig], key: str, default, composer_dict: Optional[dict[str, Any]] = None
):
    # 1) Explicit composer dict (strongest signal for custom loss settings)
    if isinstance(composer_dict, dict) and key in composer_dict and composer_dict[key] is not None:
        return composer_dict[key]

    # 2) Module-level composer config (set in-worker or loaded from env)
    _ensure_composer_config_loaded()
    if key in _COMPOSER_CONFIG and _COMPOSER_CONFIG[key] is not None:
        return _COMPOSER_CONFIG[key]

    # 3) ActorConfig / FSDPActorConfig (veRL-native keys)
    if config is not None:
        getter = getattr(config, "get", None)
        if callable(getter):
            val = getter(key, None)
            if val is not None:
                return val
        val = getattr(config, key, None)
        if val is not None:
            return val

    return default


def _config_get_context(
    config: Optional[ActorConfig], key: str, default=None, composer_dict: Optional[dict[str, Any]] = None
):
    return _config_get(config, key, default, composer_dict)


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


# ---------------------------------------------------------------------------
# Aggregation registry
# ---------------------------------------------------------------------------


def agg_token_mean(loss_mat, mask, config, **kwargs):
    return TokenMeanAggregation().aggregate(loss_mat, mask)


def agg_token_sum(loss_mat, mask, config, **kwargs):
    return TokenSumAggregation().aggregate(loss_mat, mask)


def agg_global_token(loss_mat, mask, config, **kwargs):
    return GlobalTokenAggregation().aggregate(loss_mat, mask)


def agg_group_uniform(loss_mat, mask, config, **kwargs):
    return GroupUniformAggregation().aggregate(loss_mat, mask)


def agg_trajectory_level(loss_mat, mask, config, **kwargs):
    return TrajectoryLevelAggregation().aggregate(loss_mat, mask)


def agg_weighted_token(loss_mat, mask, config, **kwargs):
    log_probs = kwargs.get("log_prob")
    if log_probs is not None:
        log_probs = _as_tensor(log_probs, name="log_prob", device=mask.device)
        _validate_tensor_shape(log_probs, ndim=(2,), first_dim=mask.shape[0], name="log_prob")
        if log_probs.shape != mask.shape:
            raise ValueError(f"log_prob shape must match mask shape, got {log_probs.shape} vs {mask.shape}")

    agg = WeightedTokenAggregation(
        alpha=_config_get(config, "tr_alpha", 1.0),
        tau=_config_get(config, "tr_tau", 1.0),
        mu=_config_get(config, "tr_mu", 0.5),
    )
    return agg.aggregate(loss_mat, mask, log_probs=log_probs)


def agg_group_learnable(loss_mat, mask, config, **kwargs):
    module = kwargs.get("composer_group_learnable_module")
    if module is None:
        # Safe fallback for formula-only runs. Learnable lambda requires trainer-side optimizer wiring.
        learnable = bool(_config_get(config, "lambda_learnable", False))
        if learnable:
            raise ValueError(
                "agg_mode=group_learnable with lambda_learnable=true requires "
                "trainer-injected 'composer_group_learnable_module'."
            )
        module = GroupLearnableAggregation(
            lambda_=float(_config_get(config, "lambda_init", 0.0)),
            r=float(_config_get(config, "lambda_r", 0.1111)),
            learnable=False,
        )

    return module.aggregate(loss_mat, mask)


def agg_difficulty_weighted(loss_mat, mask, config, **kwargs):
    module = kwargs.get("composer_difficulty_agg_module")
    learnable = bool(_config_get(config, "difficulty_weight_learnable", True))
    init_weight = float(_config_get(config, "difficulty_weight_init", 1.0))
    if module is None:
        if learnable:
            raise ValueError(
                "agg_mode=difficulty_weighted with difficulty_weight_learnable=true requires "
                "trainer-injected 'composer_difficulty_agg_module'."
            )
        module = DifficultyWeightedAggregation(
            num_bins=int(_config_get(config, "difficulty_bins", 10)),
            weight_c=float(_config_get(config, "difficulty_weight_c", 1.0)),
            learnable=False,
            init_weight=init_weight,
        )

    batch_size = mask.shape[0]

    mu_id_row = None
    for key in ("daro_mu_id_row", "composer_daro_mu_id_row", "mu_id_row", "composer_mu_id_row"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1 and candidate.shape[0] == batch_size:
            mu_id_row = candidate.to(dtype=torch.long)
            break

    inv_group_tokens_row = None
    for key in (
        "daro_inv_group_tokens_row",
        "composer_daro_inv_group_tokens_row",
        "inv_group_tokens_row",
        "composer_inv_group_tokens_row",
    ):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1 and candidate.shape[0] == batch_size:
            inv_group_tokens_row = candidate
            break

    active_mu_ids = None
    for key in ("daro_active_mu_ids", "composer_daro_active_mu_ids", "active_mu_ids", "composer_active_mu_ids"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1:
            active_mu_ids = candidate.to(dtype=torch.long)
            break

    if mu_id_row is None or inv_group_tokens_row is None:
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            shape_debug = []
            for key in (
                "daro_mu_id_row",
                "composer_daro_mu_id_row",
                "mu_id_row",
                "composer_mu_id_row",
                "daro_inv_group_tokens_row",
                "composer_daro_inv_group_tokens_row",
                "inv_group_tokens_row",
                "composer_inv_group_tokens_row",
                "daro_active_mu_ids",
                "composer_daro_active_mu_ids",
            ):
                value = kwargs.get(key)
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    shape_debug.append(f"{key}=np{tuple(value.shape)}")
                elif isinstance(value, torch.Tensor):
                    shape_debug.append(f"{key}=torch{tuple(value.shape)}")
                else:
                    shape_debug.append(f"{key}={type(value).__name__}")
            print(
                "[composer-debug][difficulty_weighted] "
                f"missing DARO row context for B={batch_size}. candidates={shape_debug}"
            )
        raise ValueError(
            "difficulty_weighted requires DARO row context from trainer. "
            "Expected daro_mu_id_row and daro_inv_group_tokens_row aligned to current microbatch."
        )

    return module.aggregate(
        loss_mat,
        mask,
        mu_id_row=mu_id_row,
        inv_group_tokens_row=inv_group_tokens_row,
        active_mu_ids=active_mu_ids,
    )


AGG_REGISTRY = {
    "token_mean": agg_token_mean,
    "token_sum": agg_token_sum,
    "global_token": agg_global_token,
    "weighted_token": agg_weighted_token,
    "group_uniform": agg_group_uniform,
    "trajectory_level": agg_trajectory_level,
    "group_learnable": agg_group_learnable,
    "difficulty_weighted": agg_difficulty_weighted,
}

VERL_AGG_MODES = {
    "verl_token_mean": "token-mean",
    "verl_seq_mean_token_sum": "seq-mean-token-sum",
    "verl_seq_mean_token_mean": "seq-mean-token-mean",
}


# ---------------------------------------------------------------------------
# Regularizer registry
# ---------------------------------------------------------------------------


def reg_none(log_prob, old_log_prob, response_mask, config, **kwargs):
    import os
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print("⚖️  [DEBUG] Regularizer: None (DAPO style, returning 0.0)")
    return torch.tensor(0.0, device=log_prob.device)


def reg_kl(log_prob, old_log_prob, response_mask, config, **kwargs):
    regularizer = KLDivergenceRegularizer()
    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        mask=response_mask,
        rewards=dummy_rewards,
    )


def reg_weighted_kl(log_prob, old_log_prob, response_mask, config, **kwargs):
    regularizer = WeightedKLDivergenceRegularizer()
    alpha = _config_get(config, "tr_alpha", 1.0)
    tau = _config_get(config, "tr_tau", 1.0)
    mu = _config_get(config, "tr_mu", 0.5)
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


def reg_preference(log_prob, old_log_prob, response_mask, config, **kwargs):
    beta_dpo = _config_get(config, "beta_dpo", 0.1)
    delta_reward = _config_get(config, "delta_reward", 0.0)
    regularizer = PreferenceRegularizer(beta_dpo=beta_dpo, delta_reward=delta_reward)

    rewards = kwargs.get("sequence_rewards")
    if rewards is None:
        rewards = kwargs.get("composer_sequence_rewards")
    if rewards is None:
        rewards = _config_get_context(config, "_composer_rewards", None)

    if rewards is None:
        raise ValueError(
            "regularizer='preference' requires sequence rewards. "
            "Provide one of: sequence_rewards, composer_sequence_rewards, or config._composer_rewards."
        )

    rewards = _as_tensor(rewards, name="sequence_rewards", device=log_prob.device)
    _validate_tensor_shape(rewards, ndim=(1,), first_dim=log_prob.shape[0], name="sequence_rewards")
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        rewards=rewards,
        mask=response_mask,
    )


def reg_mutual_info(log_prob, old_log_prob, response_mask, config, **kwargs):
    alpha = _config_get(config, "info_alpha", 1.0)
    regularizer = MutualInformationRegularizer(alpha=alpha)

    log_probs_aug = kwargs.get("log_probs_aug")
    if log_probs_aug is None:
        log_probs_aug = kwargs.get("composer_log_probs_aug")
    if log_probs_aug is None:
        log_probs_aug = _config_get_context(config, "_composer_log_probs_aug", None)

    mask_aug = kwargs.get("mask_aug")
    if mask_aug is None:
        mask_aug = kwargs.get("composer_mask_aug")
    if mask_aug is None:
        mask_aug = _config_get_context(config, "_composer_mask_aug", None)

    if log_probs_aug is None or mask_aug is None:
        missing_fields = []
        if log_probs_aug is None:
            missing_fields.append("log_probs_aug")
        if mask_aug is None:
            missing_fields.append("mask_aug")
        raise ValueError(
            "regularizer='mutual_info' requires augmented rollout tensors. "
            f"Missing: {', '.join(missing_fields)}. "
            "Provide log_probs_aug/composer_log_probs_aug and mask_aug/composer_mask_aug."
        )

    log_probs_aug = _as_tensor(log_probs_aug, name="log_probs_aug", device=log_prob.device)
    mask_aug = _as_tensor(mask_aug, name="mask_aug", device=log_prob.device)
    _validate_tensor_shape(log_probs_aug, ndim=(2,), first_dim=log_prob.shape[0], name="log_probs_aug")
    _validate_tensor_shape(mask_aug, ndim=(2,), first_dim=log_prob.shape[0], name="mask_aug")
    if log_probs_aug.shape != mask_aug.shape:
        raise ValueError(
            f"log_probs_aug and mask_aug shape mismatch: {log_probs_aug.shape} vs {mask_aug.shape}"
        )

    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        rewards=dummy_rewards,
        mask=response_mask,
        log_probs_aug=log_probs_aug,
        mask_aug=mask_aug,
    )


def reg_log_weight(log_prob, old_log_prob, response_mask, config, **kwargs):
    weights = kwargs.get("difficulty_weights")
    if weights is None:
        weights = kwargs.get("composer_difficulty_weights")
    if weights is None:
        return torch.tensor(0.0, device=log_prob.device)

    weights = _as_tensor(weights, name="difficulty_weights", device=log_prob.device)
    regularizer = LogWeightRegularizer(num_bins=weights.shape[0])
    dummy_rewards = torch.zeros(log_prob.shape[0], device=log_prob.device)
    return regularizer.compute_regularization(
        log_probs=log_prob,
        ref_log_probs=old_log_prob,
        rewards=dummy_rewards,
        mask=response_mask,
        weights=weights,
    )


REG_REGISTRY = {
    "none": reg_none,
    "kl": reg_kl,
    "weighted_kl": reg_weighted_kl,
    "preference": reg_preference,
    "mutual_info": reg_mutual_info,
    "log_weight": reg_log_weight,
}


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
