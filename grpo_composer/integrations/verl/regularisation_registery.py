"""Composer regularizer registry."""

from __future__ import annotations

import os

import torch

from grpo_composer.core.regularizers.kl_divergence import (
    KLDivergenceRegularizer,
    WeightedKLDivergenceRegularizer,
)
from grpo_composer.core.regularizers.log_weight import LogWeightRegularizer
from grpo_composer.core.regularizers.mutual_information import MutualInformationRegularizer
from grpo_composer.core.regularizers.preference import PreferenceRegularizer

from .loss_context import (
    config_get as _config_get,
    config_get_context as _config_get_context,
)
from .utils import _as_tensor, _validate_tensor_shape


# ---------------------------------------------------------------------------
# Regularizer registry
# ---------------------------------------------------------------------------


def reg_none(log_prob, old_log_prob, response_mask, config, **kwargs):
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

    # Info-GRPO Dynamic Hook Path:
    # If we are using the info_grpo hook, log_prob contains 2G sequences per prompt.
    # The first G are original (T_ori), the second G are augmented (T_aug).
    # We can perform surgery here to split them, avoiding the need for explicit augmented tensors!
    is_info_grpo = _config_get(config, "composer_flow", "") == "info_grpo"
    
    if is_info_grpo:
        # Get the total group size (e.g. 8)
        G_total = int(kwargs.get("rollout_n", kwargs.get("n", 8))) # Passed via loss_extra_kwargs
        
        # log_prob shape: (B_micro, T)
        # Because veRL uses interleave=True, the B_micro dimension is naturally grouped by prompt.
        B_micro = log_prob.shape[0]
        
        # Ensure it perfectly divides into prompts
        if B_micro % G_total != 0:
            raise ValueError(
                f"Microbatch size ({B_micro}) must be cleanly divisible by rollout.n ({G_total}) "
                "for dynamic Info-GRPO mutual info splitting."
            )
            
        B_prompts = B_micro // G_total
        G = G_total // 2
        
        # Reshape to (B_prompts, G_total, T), then split the G_total dimension in half!
        # log_prob is (B_micro, T) -> (B_prompts, 2G, T)
        lp_reshaped = log_prob.view(B_prompts, G_total, -1)
        mask_reshaped = response_mask.view(B_prompts, G_total, -1)
        
        T_dim = lp_reshaped.shape[2]
        # print(lp_reshaped)
        # print(lp_reshaped.shape)
        log_probs_ori = lp_reshaped[:, :G, :].reshape(B_prompts * G, T_dim)
        mask_ori = mask_reshaped[:, :G, :].reshape(B_prompts * G, T_dim)
        
        log_probs_aug = lp_reshaped[:, G:, :].reshape(B_prompts * G, T_dim)
        mask_aug = mask_reshaped[:, G:, :].reshape(B_prompts * G, T_dim)
        
        # Replace the active tensors with ONLY the ori halves for the computation, 
        # so the regularizer sees exactly G trajectories!
        active_log_prob = log_probs_ori
        active_response_mask = mask_ori
        active_old_log_prob = old_log_prob.view(B_prompts, G_total, T_dim)[:, :G, :].reshape(B_prompts * G, T_dim)
        
    else:
        # Legacy Path: Explicit Tensors
        active_log_prob = log_prob
        active_response_mask = response_mask
        active_old_log_prob = old_log_prob
        
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
            raise ValueError(
                "regularizer='mutual_info' requires augmented rollout tensors natively or via the info_grpo hook. "
                "Provide log_probs_aug and mask_aug."
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
