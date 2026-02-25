"""
veRL Advantage Estimator Registrations

Registers grpo_composer's custom advantage estimators into veRL's
ADV_ESTIMATOR_REGISTRY so they can be selected via config:
    algorithm.adv_estimator: "difficulty_aware_grpo"

All functions follow veRL's expected signature:
    (token_level_rewards, response_mask, index, epsilon, config, **kwargs)
    → (advantages, returns)

Where:
    - token_level_rewards: (bs, response_length)
    - response_mask:       (bs, response_length)
    - index:               (bs,) np.ndarray — group ID per sample
    - Returns are both     (bs, response_length)
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.config import AlgoConfig


# ──────────────────────────────────────────────────────────────
# 1. Difficulty-Aware Advantage (GRPO-LEAD)
# ──────────────────────────────────────────────────────────────
@register_adv_est("difficulty_aware_grpo")
def compute_difficulty_aware_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO-LEAD: Difficulty-aware reweighting.
    Upweights hard prompts (low accuracy) and downweights easy ones.

    Config keys:
        difficulty_A, difficulty_B: logistic bounds (default 0.2, 1.0)
        difficulty_k: steepness (default 10.0)
        difficulty_rho_0: midpoint (default 0.5)
    """
    A = config.get("difficulty_A", 0.2) if config else 0.2
    B = config.get("difficulty_B", 1.0) if config else 1.0
    k = config.get("difficulty_k", 10.0) if config else 10.0
    rho_0 = config.get("difficulty_rho_0", 0.5) if config else 0.5

    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        id2scores = defaultdict(list)
        id2indices = defaultdict(list)
        bsz = scores.shape[0]

        for i in range(bsz):
            id2scores[index[i]].append(scores[i])
            id2indices[index[i]].append(i)

        result = torch.zeros_like(scores)

        for idx in id2scores:
            group_scores = torch.stack(id2scores[idx])
            group_idx = id2indices[idx]
            G = len(group_scores)

            mean = group_scores.mean()
            std = group_scores.std() if G > 1 else torch.tensor(1.0)

            # Difficulty proxy: fraction of "correct" (positive reward)
            rho = (group_scores > 0).float().mean()

            # Logistic weight
            w = A + (B - A) / (1 + torch.exp(k * (rho - rho_0)))

            for j, i in enumerate(group_idx):
                base_adv = (group_scores[j] - mean) / (std + epsilon)
                if base_adv > 0:
                    result[i] = base_adv * w
                else:
                    result[i] = base_adv * (1 - w)

        advantages = result.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 2. Length-Corrected Advantage (TIC-GRPO)
# ──────────────────────────────────────────────────────────────
@register_adv_est("length_corrected_grpo")
def compute_length_corrected_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TIC-GRPO: Normalize rewards by sequence length before advantage.
    Removes dependence on sequence length variance.
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    lengths = response_mask.sum(dim=-1).clamp(min=1)  # (bs,)
    reward_per_token = scores / lengths

    with torch.no_grad():
        id2rpt = defaultdict(list)
        bsz = reward_per_token.shape[0]

        for i in range(bsz):
            id2rpt[index[i]].append(reward_per_token[i])

        id2mean, id2std = {}, {}
        for idx in id2rpt:
            group = torch.stack(id2rpt[idx])
            id2mean[idx] = group.mean()
            id2std[idx] = group.std() if len(group) > 1 else torch.tensor(1.0)

        for i in range(bsz):
            reward_per_token[i] = (reward_per_token[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        advantages = reward_per_token.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 3. Kalman Filter Advantage (KRPO)
# ──────────────────────────────────────────────────────────────
# Stateful — needs a closure to maintain Kalman state across steps
_kalman_state = {"x_hat": None, "P": None}


@register_adv_est("kalman_grpo")
def compute_kalman_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KRPO: Kalman filter baseline instead of group mean.
    Provides a dynamic, smoothed baseline estimate.

    Config keys:
        kalman_Q: process noise (default 1e-4)
        kalman_R: measurement noise (default 1.0)
    """
    Q = config.get("kalman_Q", 1e-4) if config else 1e-4
    R = config.get("kalman_R", 1.0) if config else 1.0

    scores = token_level_rewards.sum(dim=-1)
    batch_mean = scores.mean().item()

    with torch.no_grad():
        # Kalman update
        if _kalman_state["x_hat"] is None:
            _kalman_state["x_hat"] = batch_mean
            _kalman_state["P"] = 1.0
        else:
            x_pred = _kalman_state["x_hat"]
            P_pred = _kalman_state["P"] + Q
            K = P_pred / (P_pred + R)
            _kalman_state["x_hat"] = x_pred + K * (batch_mean - x_pred)
            _kalman_state["P"] = (1 - K) * P_pred

        x_hat = _kalman_state["x_hat"]
        P = _kalman_state["P"]

        advantages = (scores - x_hat) / (P ** 0.5 + epsilon)
        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 4. Decoupled Multi-Reward Advantage (GDPO)
# ──────────────────────────────────────────────────────────────
@register_adv_est("decoupled_grpo")
def compute_decoupled_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO: Per-reward-type normalization before aggregation.
    Prevents dominant rewards from drowning weaker signals.

    Expects multi-reward scores passed via kwargs["multi_rewards"] as (bs, K).
    Falls back to standard GRPO if not provided.
    """
    multi_rewards = kwargs.get("multi_rewards", None)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        if multi_rewards is not None:
            # multi_rewards: (bs, K) — K reward types
            # Normalize each reward type within groups
            K = multi_rewards.shape[1]
            normalized = torch.zeros_like(multi_rewards)

            for k_idx in range(K):
                col = multi_rewards[:, k_idx]
                id2vals = defaultdict(list)
                bsz = col.shape[0]
                for i in range(bsz):
                    id2vals[index[i]].append(col[i])

                for idx in id2vals:
                    group = torch.stack(id2vals[idx])
                    mean = group.mean()
                    std = group.std() if len(group) > 1 else torch.tensor(1.0)
                    for i, gidx in enumerate([j for j in range(bsz) if index[j] == idx]):
                        normalized[gidx, k_idx] = (col[gidx] - mean) / (std + epsilon)

            # Sum normalized advantages
            combined = normalized.sum(dim=-1)
            # Batch-level normalization
            combined = (combined - combined.mean()) / (combined.std() + epsilon)
            advantages = combined.unsqueeze(-1) * response_mask
        else:
            # Fallback: standard GRPO
            id2score = defaultdict(list)
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            id2mean, id2std = {}, {}
            for idx in id2score:
                group = torch.stack(id2score[idx])
                id2mean[idx] = group.mean()
                id2std[idx] = group.std() if len(group) > 1 else torch.tensor(1.0)

            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

            advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 5. Multi-Scale Advantage (MS-GRPO)
# ──────────────────────────────────────────────────────────────
@register_adv_est("multi_scale_grpo")
def compute_multi_scale_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MS-GRPO: Hierarchical subgroup advantage at multiple scales.
    Averages advantages computed over all subgroups of sizes τ_min..G.

    Config keys:
        ms_tau_min: minimum subgroup size (default 2)
    """
    from itertools import combinations

    tau_min = config.get("ms_tau_min", 2) if config else 2
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        id2scores = defaultdict(list)
        id2indices = defaultdict(list)
        bsz = scores.shape[0]

        for i in range(bsz):
            id2scores[index[i]].append(scores[i])
            id2indices[index[i]].append(i)

        result = torch.zeros_like(scores)

        for idx in id2scores:
            group_scores = torch.stack(id2scores[idx])
            group_idx = id2indices[idx]
            G = len(group_scores)

            if G < 2:
                continue

            tau_max = G
            for j, global_i in enumerate(group_idx):
                scale_advs = []
                for tau in range(max(tau_min, 2), tau_max + 1):
                    others = [k for k in range(G) if k != j]
                    subgroup_advs = []
                    for combo in combinations(others, tau - 1):
                        sub_idx = [j] + list(combo)
                        sub_rewards = group_scores[sub_idx]
                        mean = sub_rewards.mean()
                        std = sub_rewards.std() + epsilon
                        subgroup_advs.append((group_scores[j] - mean) / std)
                    if subgroup_advs:
                        scale_advs.append(torch.stack(subgroup_advs).mean())

                if scale_advs:
                    # Uniform weights across scales
                    result[global_i] = torch.stack(scale_advs).mean()

        advantages = result.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 6. Static Value Advantage (PVPO)
# ──────────────────────────────────────────────────────────────
@register_adv_est("static_value_grpo")
def compute_static_value_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PVPO: Uses static reference baseline instead of dynamic group mean.

    Expects reference rewards via kwargs["reference_rewards"] as (bs,).
    Falls back to group mean if not provided.
    """
    scores = token_level_rewards.sum(dim=-1)
    ref_rewards = kwargs.get("reference_rewards", None)

    with torch.no_grad():
        if ref_rewards is not None:
            # PVPO: subtract reference mean per group
            id2ref = defaultdict(list)
            bsz = scores.shape[0]
            for i in range(bsz):
                id2ref[index[i]].append(ref_rewards[i])

            for i in range(bsz):
                ref_mean = torch.stack(id2ref[index[i]]).mean()
                scores[i] = scores[i] - ref_mean
        else:
            # Fallback to standard group mean subtraction
            id2score = defaultdict(list)
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            id2mean = {}
            for idx in id2score:
                id2mean[idx] = torch.stack(id2score[idx]).mean()

            for i in range(bsz):
                scores[i] = scores[i] - id2mean[index[i]]

        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


# ──────────────────────────────────────────────────────────────
# 7. Novelty-Sharpened Advantage (XRPO)
# ──────────────────────────────────────────────────────────────
@register_adv_est("novelty_sharp_grpo")
def compute_novelty_sharpened_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    XRPO: Boosts advantages for novel (low-likelihood) yet correct sequences.

    Requires kwargs["old_log_probs"] (bs, response_length) for computing
    per-sequence log-likelihood scores.

    Config keys:
        novelty_lambda: bonus scaling (default 1.0)
        novelty_kappa: clipping for bonus (default 1.0)
    """
    lam = config.get("novelty_lambda", 1.0) if config else 1.0
    kappa = config.get("novelty_kappa", 1.0) if config else 1.0

    old_log_probs = kwargs.get("old_log_probs", None)
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        # Standard GRPO advantage first
        id2score = defaultdict(list)
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        id2mean, id2std = {}, {}
        for idx in id2score:
            group = torch.stack(id2score[idx])
            id2mean[idx] = group.mean()
            id2std[idx] = group.std() if len(group) > 1 else torch.tensor(1.0)

        base_adv = torch.zeros_like(scores)
        for i in range(bsz):
            base_adv[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # Novelty bonus (only if log_probs available)
        if old_log_probs is not None:
            # Per-sequence mean log-likelihood
            seq_llk = (old_log_probs * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)

            # Group-level novelty
            id2llk = defaultdict(list)
            for i in range(bsz):
                id2llk[index[i]].append(seq_llk[i])

            eta = torch.zeros_like(scores)
            for i in range(bsz):
                group_mean_llk = torch.stack(id2llk[index[i]]).mean()
                eta[i] = torch.exp(seq_llk[i] - group_mean_llk)

            novelty_bonus = lam * (1 - eta)
            novelty_bonus = torch.clamp(novelty_bonus, min=0)
            novelty_bonus = torch.min(novelty_bonus, kappa * base_adv.abs())

            base_adv = base_adv + novelty_bonus

        advantages = base_adv.unsqueeze(-1) * response_mask

    return advantages, advantages
