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

All sub-components import from grpo_composer.core/advantages/ — the
integration layer is a thin adapter translating veRL's flat (bs, T)
tensors + index array into core/'s grouped (B, G) format.

Shape Mapping:
    veRL gives:   (bs, response_length) flat, with index array for grouping
    core/ expects: (B, G) grouped rewards
    Adapter: groups by index → reshapes to (B, G) → calls core/ → ungrouped back to (bs,)
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.config import AlgoConfig

# ── Import YOUR core advantage implementations ──
from grpo_composer.core.advantages.standard import StandardAdvantageFunction
from grpo_composer.core.advantages.unbiased import UnbiasedAdvantageFunction
from grpo_composer.core.advantages.difficulty_aware import DifficultyAwareAdvantageFunction
from grpo_composer.core.advantages.length_corrected import LengthCorrectedAdvantageFunction
from grpo_composer.core.advantages.kalman import KalmanAdvantageFunction
from grpo_composer.core.advantages.decoupled import DecoupledAdvantageFunction
from grpo_composer.core.advantages.multi_scale import MultiScaleAdvantageFunction
from grpo_composer.core.advantages.static_value import StaticValueAdvantageFunction
from grpo_composer.core.advantages.novelty_sharpening import NoveltySharpeningAdvantageFunction


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _group_scores(token_level_rewards, response_mask, index):
    """
    Convert veRL's flat (bs, response_length) + index array
    into grouped rewards (B, G) for core/ advantage functions.

    Returns:
        grouped_rewards: (B, G) tensor
        group_keys: list of unique group IDs (length B)
        group_indices: dict mapping group_id → list of flat indices
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,) — per-sequence score
    bsz = scores.shape[0]

    group_indices = defaultdict(list)
    for i in range(bsz):
        group_indices[index[i]].append(i)

    group_keys = sorted(group_indices.keys())
    G = max(len(v) for v in group_indices.values())

    # Build (B, G) tensor — pad groups smaller than G
    grouped = torch.zeros(len(group_keys), G, device=scores.device)
    for b, key in enumerate(group_keys):
        indices = group_indices[key]
        for g, idx in enumerate(indices):
            grouped[b, g] = scores[idx]

    return grouped, group_keys, group_indices, scores


def _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask):
    """
    Convert grouped advantages (B, G) back to veRL's flat (bs, response_length).
    Broadcasts the per-sequence advantage across valid tokens.
    """
    result = torch.zeros(bsz, device=grouped_adv.device)
    for b, key in enumerate(group_keys):
        indices = group_indices[key]
        for g, idx in enumerate(indices):
            result[idx] = grouped_adv[b, g]

    # Broadcast to (bs, response_length) — same advantage for all tokens in sequence
    advantages = result.unsqueeze(-1) * response_mask
    return advantages, advantages


# ═══════════════════════════════════════════════════════════════
# 1. Difficulty-Aware Advantage (GRPO-LEAD)
#    Wraps core/advantages/difficulty_aware.py
# ═══════════════════════════════════════════════════════════════
@register_adv_est("difficulty_aware_grpo")
def compute_difficulty_aware_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wraps core/advantages/difficulty_aware.py — DifficultyAwareAdvantageFunction"""
    A = config.get("difficulty_A", 0.2) if config else 0.2
    B_param = config.get("difficulty_B", 1.0) if config else 1.0
    k = config.get("difficulty_k", 10.0) if config else 10.0
    rho_0 = config.get("difficulty_rho_0", 0.5) if config else 0.5

    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    with torch.no_grad():
        fn = DifficultyAwareAdvantageFunction(A=A, B=B_param, k=k, rho_0=rho_0, epsilon=epsilon)
        grouped_adv = fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 2. Length-Corrected Advantage (TIC-GRPO)
#    Wraps core/advantages/length_corrected.py
# ═══════════════════════════════════════════════════════════════
@register_adv_est("length_corrected_grpo")
def compute_length_corrected_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wraps core/advantages/length_corrected.py — LengthCorrectedAdvantageFunction"""
    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    # Build grouped lengths (B, G) from response_mask
    lengths = response_mask.sum(dim=-1)  # (bs,)
    G = grouped.shape[1]
    grouped_lengths = torch.ones_like(grouped)
    for b, key in enumerate(group_keys):
        indices = group_indices[key]
        for g, idx in enumerate(indices):
            grouped_lengths[b, g] = lengths[idx]

    with torch.no_grad():
        fn = LengthCorrectedAdvantageFunction(epsilon=epsilon)
        grouped_adv = fn.compute_advantages(grouped, grouped_lengths)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 3. Kalman Filter Advantage (KRPO)
#    Wraps core/advantages/kalman.py
# ═══════════════════════════════════════════════════════════════
# Persistent instance to maintain filter state across steps
_kalman_fn = None


@register_adv_est("kalman_grpo")
def compute_kalman_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wraps core/advantages/kalman.py — KalmanAdvantageFunction (stateful)"""
    global _kalman_fn
    Q = config.get("kalman_Q", 1e-4) if config else 1e-4
    R = config.get("kalman_R", 1.0) if config else 1.0

    if _kalman_fn is None:
        _kalman_fn = KalmanAdvantageFunction(process_noise=Q, measurement_noise=R, epsilon=epsilon)

    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    with torch.no_grad():
        grouped_adv = _kalman_fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 4. Decoupled Multi-Reward Advantage (GDPO)
#    Wraps core/advantages/decoupled.py
# ═══════════════════════════════════════════════════════════════
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
    Wraps core/advantages/decoupled.py — DecoupledAdvantageFunction

    Expects kwargs["multi_rewards"] as (bs, K) for K reward types.
    Falls back to standard GRPO if not provided.
    """
    multi_rewards = kwargs.get("multi_rewards", None)
    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    with torch.no_grad():
        if multi_rewards is not None:
            # Build grouped multi-rewards (B, G, K)
            K = multi_rewards.shape[1]
            G = grouped.shape[1]
            grouped_multi = torch.zeros(len(group_keys), G, K, device=scores.device)
            for b, key in enumerate(group_keys):
                indices = group_indices[key]
                for g, idx in enumerate(indices):
                    grouped_multi[b, g] = multi_rewards[idx]

            fn = DecoupledAdvantageFunction(eps=epsilon)
            grouped_adv = fn.compute_advantages(grouped_multi)
        else:
            # Fallback: standard GRPO
            fn = StandardAdvantageFunction()
            grouped_adv = fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 5. Multi-Scale Advantage (MS-GRPO)
#    Wraps core/advantages/multi_scale.py
# ═══════════════════════════════════════════════════════════════
@register_adv_est("multi_scale_grpo")
def compute_multi_scale_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wraps core/advantages/multi_scale.py — MultiScaleAdvantageFunction"""
    tau_min = config.get("ms_tau_min", 2) if config else 2

    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    with torch.no_grad():
        fn = MultiScaleAdvantageFunction(tau_min=tau_min, epsilon=epsilon)
        grouped_adv = fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 6. Static Value Advantage (PVPO)
#    Wraps core/advantages/static_value.py
# ═══════════════════════════════════════════════════════════════
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
    Wraps core/advantages/static_value.py — StaticValueAdvantageFunction

    Expects kwargs["reference_rewards"] as (bs,) for static baseline.
    Falls back to standard group mean if not provided.
    """
    ref_rewards = kwargs.get("reference_rewards", None)
    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]

    with torch.no_grad():
        if ref_rewards is not None:
            # Build grouped reference rewards (B, G)
            G = grouped.shape[1]
            grouped_ref = torch.zeros_like(grouped)
            for b, key in enumerate(group_keys):
                indices = group_indices[key]
                for g, idx in enumerate(indices):
                    grouped_ref[b, g] = ref_rewards[idx]

            fn = StaticValueAdvantageFunction(epsilon=epsilon)
            grouped_adv = fn.compute_advantages(grouped, grouped_ref)
        else:
            fn = StandardAdvantageFunction()
            grouped_adv = fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)


# ═══════════════════════════════════════════════════════════════
# 7. Novelty-Sharpened Advantage (XRPO)
#    Wraps core/advantages/novelty_sharpening.py
# ═══════════════════════════════════════════════════════════════
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
    Wraps core/advantages/novelty_sharpening.py — NoveltySharpeningAdvantageFunction

    Requires kwargs["old_log_probs"] (bs, response_length) for computing
    per-sequence log-likelihood novelty scores.

    Note: Core function expects log_probs as (B, G, T) and internally computes
    s = log_probs.mean(dim=-1) → (B, G). We build the full (B, G, T) grouped
    tensor from veRL's flat (bs, response_length).
    """
    lam = config.get("novelty_lambda", 1.0) if config else 1.0
    kappa = config.get("novelty_kappa", 1.0) if config else 1.0

    old_log_probs = kwargs.get("old_log_probs", None)
    grouped, group_keys, group_indices, scores = _group_scores(
        token_level_rewards, response_mask, index
    )
    bsz = scores.shape[0]
    T = token_level_rewards.shape[1]

    with torch.no_grad():
        if old_log_probs is not None:
            # Build grouped log-probs (B, G, T) from flat (bs, T)
            # Core function expects full (B, G, T) — it does mean(dim=-1) internally
            B_groups = len(group_keys)
            G = grouped.shape[1]
            grouped_llk = torch.zeros(B_groups, G, T, device=scores.device)
            for b, key in enumerate(group_keys):
                indices = group_indices[key]
                for g, idx in enumerate(indices):
                    grouped_llk[b, g] = old_log_probs[idx]

            fn = NoveltySharpeningAdvantageFunction(
                lambda_novelty=lam, kappa_clip=kappa, epsilon=epsilon
            )
            grouped_adv = fn.compute_advantages(grouped, grouped_llk)
        else:
            # Fallback: standard without novelty bonus
            fn = StandardAdvantageFunction()
            grouped_adv = fn.compute_advantages(grouped)

    return _ungroup_advantages(grouped_adv, group_keys, group_indices, bsz, response_mask)
