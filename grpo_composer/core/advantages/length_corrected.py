"""
TIC-GRPO: Trajectory-level Importance-Corrected, Length-Corrected Group Normalization

Paper: TIC-GRPO

Components Changed (from base GRPO):
- Reward normalized by sequence length BEFORE advantage calculation
- Better convergence rate without σ²_{sT} terms

Mathematical Form:
    Standard GRPO:
        A_G(s_T) = (r(s_T) - μ_G) / (σ_G + δ)

    TIC-GRPO (length-corrected):
        A'_G(s_T) = (r(s_T) / |s_T| - μ'_G) / (σ'_G + δ)

    Where μ'_G and σ'_G are computed on length-normalized rewards.

Benefit:
    Removes dependence on sequence length variance σ²_{sT,N}
"""

import torch
from .base import AdvantageFunction


class LengthCorrectedAdvantageFunction(AdvantageFunction):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        TIC-GRPO advantage computation (length-corrected).

        Args:
            rewards: (B, G) binary rewards per trajectory
            lengths: (B, G) number of tokens per trajectory

        Returns:
            advantages: (B, G) length-corrected advantages

        Shape Flow:
            rewards: (B, G), lengths: (B, G)
            reward_per_token: (B, G) — r / |o|
            mean: (B, 1) — group mean of reward_per_token
            std: (B, 1) — group std of reward_per_token
            advantages: (B, G) — (reward_per_token - mean) / (std + ε)
        """
        # 1. Convert reward -> reward per token
        reward_per_token = rewards / (lengths + self.epsilon)

        # 2. Group normalization
        mean = reward_per_token.mean(dim=-1, keepdim=True)
        std = reward_per_token.std(dim=-1, keepdim=True) + self.epsilon

        advantages = (reward_per_token - mean) / std
        return advantages
