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
import torch.nn as nn
import torch.nn.functional as F

class NoveltySharpeningAdvantageFunction(AdvantageFunction):
    def __init__(self, lambda_novelty: float = 1.0, kappa_clip: float = 1.0, epsilon: float = 1e-8):
        super().__init__()
        self.lambda_novelty = lambda_novelty
        self.kappa_clip = kappa_clip
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        B, G = rewards.shape
        B, G, T = log_probs.shape

        assert B == log_probs.shape[0], "Batch size mismatch"
        assert G == log_probs.shape[1], "Group size mismatch"
        assert B is not None and G is not None and T is not None, "All dimensions must be specified"
        
        s = log_probs.mean(dim=-1) # shape : (B, G)
        s_bar = s.mean(dim=-1) # shape : (B)

        eta = torch.exp(s - s_bar)
        base_advantage = rewards - rewards.mean(dim=-1, keepdim=True)/rewards.std(dim=-1, keepdim=True)
        adjusted_advantage = base_advantage + min(max(self.lambda_novelty * (1 - eta), 0), self.kappa_clip * base_advantage)
        return adjusted_advantage
