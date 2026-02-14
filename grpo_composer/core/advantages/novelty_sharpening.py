"""
XRPO: Novelty-Guided Advantage Sharpening for Exploitation

Paper: XRPO

Components Changed (from base GRPO):
- Adds novelty bonus to advantage for novel yet correct sequences
- Expands policy boundary by boosting rare correct solutions

Mathematical Form:
    Log-likelihood score:
        s(y) = (1/|y|) * Σ_t log π_θ(y_t | x, y_{<t})

    Novelty (relative to group average):
        η_i = exp(s(y_i) - s̄)

    Advantage sharpening:
        A⁺_i = A_i + min(max(λ_novelty * (1 - η_i), 0), κ_clip * A_i)

Effect:
    Boosts novel yet correct sequences, expands policy boundary
"""

from .base import AdvantageFunction
import torch


class NoveltySharpeningAdvantageFunction(AdvantageFunction):
    """
    XRPO novelty-guided advantage sharpening.
    
    Boosts advantages for novel (low-likelihood) yet correct sequences.
    """
    
    def __init__(self, lambda_novelty: float = 1.0, kappa_clip: float = 1.0, epsilon: float = 1e-8):
        super().__init__()
        self.lambda_novelty = lambda_novelty
        self.kappa_clip = kappa_clip
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (B, G) rewards
            log_probs: (B, G, T) per-token log probabilities
            
        Returns:
            advantages: (B, G) novelty-sharpened advantages
        """
        B, G = rewards.shape
        
        # Log-likelihood score: s(y) = (1/|y|) * Σ_t log π(y_t|...)
        s = log_probs.mean(dim=-1)  # (B, G)
        
        # Group average (keepdim for broadcasting)
        s_bar = s.mean(dim=-1, keepdim=True)  # (B, 1)

        # Novelty: η_i = exp(s(y_i) - s̄)
        # Low-likelihood (novel) → s < s̄ → η < 1
        # High-likelihood (common) → s > s̄ → η > 1
        eta = torch.exp(s - s_bar)  # (B, G)
        
        # Base advantage: (r - μ) / σ
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True) + self.epsilon
        base_advantage = (rewards - mean) / std
        
        # Novelty bonus: λ * (1 - η)
        # Novel (η < 1) → positive bonus
        # Common (η > 1) → negative (but clipped to 0)
        novelty_bonus = self.lambda_novelty * (1 - eta)
        
        # Clamp: min(max(bonus, 0), κ * A)
        # Bonus only for novel sequences (clamp at 0)
        # Also cap at κ * A to prevent excessive boosting
        clamped_bonus = torch.clamp(novelty_bonus, min=0)
        clamped_bonus = torch.min(clamped_bonus, self.kappa_clip * base_advantage.abs())
        
        # A⁺ = A + bonus
        adjusted_advantage = base_advantage + clamped_bonus
        
        return adjusted_advantage
