"""
PVPO: Static Value Estimate from Reference Policy

Paper: PVPO

Components Changed (from base GRPO):
- Baseline: Uses STATIC reference mean instead of dynamic group mean
- Decouples Q (from on-policy) and V (from reference)

Mathematical Form:
    Standard GRPO (dynamic baseline):
        Â = r_i - mean(r)              # Dynamic group mean

    PVPO (static baseline):
        Â^{PVPO} = r_i - mean(r_ref)   # Static reference mean

    Where:
        Q (dynamic): From on-policy π_θ rollout reward
        V (static): Pre-estimated from reference π_ref: V̂_sta = (1/M) * Σ_j r^{ref}_j

Benefit:
    Stable learning signal even under severe reward sparsity
"""

import torch
from .base import AdvantageFunction


class StaticValueAdvantageFunction(AdvantageFunction):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor, reference_rewards: torch.Tensor) -> torch.Tensor:
        """
        PVPO advantage: reward minus static reference baseline.

        Args:
            rewards: (B, G) on-policy rewards
            reference_rewards: (B, G) pre-computed reference rewards

        Returns:
            advantages: (B, G) static-value advantages

        Shape Flow:
            rewards: (B, G)
            reference_rewards: (B, G)
            reference_mean: (B, 1) — mean over G dimension
            advantages: (B, G) — rewards - reference_mean
        """
        return rewards - reference_rewards.mean(dim=-1, keepdim=True)