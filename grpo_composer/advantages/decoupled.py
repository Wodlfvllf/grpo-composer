"""
GDPO: Decoupled Per-Reward Normalization for Multi-Reward RL

Paper: GDPO prevents reward collapse in multi-reward settings by normalizing
each reward type separately BEFORE summing, rather than summing first then normalizing.

Mathematical Form:
------------------
Standard GRPO (problematic):
    A_sum = (r_sum - μ) / σ    # Sum rewards first, then normalize

GDPO Step 1 - Normalize each reward separately:
    A_k = (r_k - mean{r_k}) / std{r_k}

GDPO Step 2 - Sum normalized advantages:
    A_sum = Σ_k A_k

GDPO Step 3 - Batch-wise normalization:
    Â_sum = (A_sum - μ_batch) / (σ_batch + ε)

Optional: Weighted aggregation
    A_sum = Σ_k w_k * A_k

Why This Matters:
- Prevents dominant rewards from drowning out weaker signals
- Each reward type contributes proportionally to gradients
- Better for RL with multiple reward components (format + correctness + thinking)
"""

import torch
from .base import AdvantageFunction


class DecoupledAdvantageFunction(AdvantageFunction):
    """
    GDPO-style decoupled advantage for multi-reward RL.
    
    Normalizes each reward component separately before aggregation
    to prevent reward collapse.
    """
    
    def __init__(self, weights: torch.Tensor = None, eps: float = 1e-8):
        """
        Args:
            weights: Optional (K,) weights for each reward type
            eps: Numerical stability constant
        """
        super().__init__()
        self.weights = weights
        self.eps = eps
    
    def compute_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute decoupled advantages for multi-reward.
        
        Args:
            rewards: (B, G, K) rewards where K = number of reward types
                     OR (B, G) for single reward (falls back to standard)
            
        Returns:
            advantages: (B, G) aggregated advantages
        """
        if rewards.dim() == 2:
            # Single reward - standard normalization
            mean = rewards.mean(dim=-1, keepdim=True)
            std = rewards.std(dim=-1, keepdim=True)
            return (rewards - mean) / (std + self.eps)
        
        # Multi-reward: (B, G, K)
        B, G, K = rewards.shape
        
        # Step 1: Normalize each reward type separately across group
        means = rewards.mean(dim=1, keepdim=True)  # (B, 1, K)
        stds = rewards.std(dim=1, keepdim=True)    # (B, 1, K)
        normalized = (rewards - means) / (stds + self.eps)  # (B, G, K)
        
        # Step 2: Weighted sum (or uniform if no weights)
        if self.weights is not None:
            advantages = (normalized * self.weights).sum(dim=-1)  # (B, G)
        else:
            advantages = normalized.sum(dim=-1)  # (B, G)
        
        # Step 3: Batch-wise normalization
        batch_mean = advantages.mean(dim=-1, keepdim=True)
        batch_std = advantages.std(dim=-1, keepdim=True)
        
        return (advantages - batch_mean) / (batch_std + self.eps)