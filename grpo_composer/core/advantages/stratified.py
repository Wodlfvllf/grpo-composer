"""
Stratified-GRPO: Per-Stratum Advantage Normalization (SAN)

Paper: "Stratified GRPO: Handling Structural Heterogeneity in RL of LLM Search Agents"

Components Changed (from base GRPO):
- Partitions trajectories into strata by structure (e.g., search count)
- Normalizes advantages WITHIN each stratum, not globally

Mathematical Form:
    Global advantage (standard GRPO):
        A^{GN}(τ_i) = (R_i - R̄_global) / (σ_global + ε)

    Stratified advantage (SAN):
        A^{SAN}(τ_i) = (R_i - μ̃_k) / (σ̃_k + ε)
        where μ̃_k, σ̃_k computed within stratum k only

    Blended (for finite-sample stability):
        A^{blend} = α · A^{SAN} + (1 - α) · A^{GN}

    Advantage decomposition:
        A^{GN} = A^{SAN} + (R̄_k - R̄_global)  [cross-stratum bias]

Benefits:
    - Eliminates between-stratum variance
    - Zero conditional mean within each stratum
    - Unit conditional variance within each stratum
    - Invariant to positive affine reward transforms
"""

import torch
from .base import AdvantageFunction


class StratifiedAdvantageFunction(AdvantageFunction):
    """
    Stratified-GRPO: Stratified Advantage Normalization (SAN).

    Args:
        alpha: Blending coefficient (1 = pure SAN, 0 = pure GN). Default 1.0.
        epsilon: Numerical stability constant. Default 1e-8.
    """

    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        strata: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute blended stratified advantages.

        Args:
            rewards: (B, G) rewards per trajectory
            strata: (B, G) integer stratum IDs per trajectory

        Returns:
            advantages: (B, G) blended SAN + GN advantages
        """
        B, G = rewards.shape

        # Global normalization (GN) — standard GRPO
        global_mean = rewards.mean(dim=-1, keepdim=True)
        global_std = rewards.std(dim=-1, keepdim=True) + self.epsilon
        a_gn = (rewards - global_mean) / global_std

        # Stratified normalization (SAN) — per-stratum
        a_san = torch.zeros_like(rewards)

        for b in range(B):
            r = rewards[b]          # (G,)
            s = strata[b]           # (G,)

            for s_id in s.unique().tolist():
                mask = (s == s_id)
                stratum_rewards = r[mask]

                mu_k = stratum_rewards.mean()
                sigma_k = stratum_rewards.std() + self.epsilon
                a_san[b, mask] = (stratum_rewards - mu_k) / sigma_k

        # Blended advantage
        return self.alpha * a_san + (1.0 - self.alpha) * a_gn