"""
RANKING IS REWARD: Advantage Clipping Based on Correctness

Paper: RankGRPO

Components Changed (from base GRPO):
- Clips advantage values based on sample correctness
- Prevents extreme gradient updates

Mathematical Form:
    Advantage clipping:
        A^{clip}_i = max(A_i, ξ⁻)   if correct
                   = min(A_i, ξ⁺)   if incorrect

    Where:
        ξ⁻ = lower bound for correct samples (prevents too-negative advantages)
        ξ⁺ = upper bound for incorrect samples (prevents too-positive advantages)

Note: This is SEPARATE from ratio clipping (ε clipping in PPO/GRPO).
      This clips the advantage values themselves.
"""

import torch
from .base import AdvantageFunction


class AdvantageClipping(AdvantageFunction):
    """
    Clips advantage values based on correctness to prevent rank-induced misalignment.
    """
    
    def __init__(self, xi_minus: float = -0.5, xi_plus: float = 0.5, eps: float = 1e-8):
        """
        Args:
            xi_minus: Lower bound for correct samples (prevents too-negative advantages)
            xi_plus: Upper bound for incorrect samples (prevents too-positive advantages)
            eps: Numerical stability constant
        """
        super().__init__()
        self.xi_minus = xi_minus
        self.xi_plus = xi_plus
        self.eps = eps
    
    def compute_advantages(self, rank_enhanced_rewards: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute clipped advantages.
        
        Args:
            rank_enhanced_rewards: (B, G) rank-enhanced rewards from RankEnhancedRewardCalculator
            rewards: (B, G) original binary rewards (0 = incorrect, >0 = correct)
            
        Returns:
            advantages: (B, G) clipped advantages
        """
        mean = rank_enhanced_rewards.mean(dim=-1, keepdim=True)
        std = rank_enhanced_rewards.std(dim=-1, keepdim=True)

        advantages = (rank_enhanced_rewards - mean) / (std + self.eps)
        
        # Clip based on correctness (vectorized)
        # Correct samples: max(A, ξ⁻) - floor prevents too-negative
        # Incorrect samples: min(A, ξ⁺) - ceiling prevents too-positive
        correct_mask = rewards > 0
        clipped = torch.where(
            correct_mask,
            torch.clamp(advantages, min=self.xi_minus),  # correct: floor at ξ⁻
            torch.clamp(advantages, max=self.xi_plus)    # incorrect: ceiling at ξ⁺
        )

        return clipped
