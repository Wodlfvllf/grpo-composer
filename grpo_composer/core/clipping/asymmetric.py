
"""
Asymmetric Clipping Mechanism for GRPO (DAPO / P-GRPO)

Asymmetric clipping with different bounds on each side:
    clip(ρ, 1-ε_l, 1+ε_h)

From DAPO paper:
- ε_l = 0.2 (lower bound, same as standard)
- ε_h = 0.28 (higher upper bound)

Rationale: Allows more aggressive probability increases for good actions,
which helps prevent entropy collapse.

Input:
------
- probs_ratio: torch.Tensor, shape (B, G, T) - token-level probability ratios

Output:
-------
- clipped_ratio: torch.Tensor, shape (B, G, T)
"""

import torch
from .base import ClippingMechanism


class AsymmetricClippingMechanism(ClippingMechanism):
    """
    DAPO/P-GRPO asymmetric clipping.
    
    clip(ρ, 1-ε_l, 1+ε_h) with different ε values for upper/lower bounds.
    Higher upper bound allows more aggressive probability increases.
    """
    
    def __init__(self, epsilon_lower: float = 0.2, epsilon_upper: float = 0.28):
        """
        Args:
            epsilon_lower: Lower bound offset (default 0.2)
            epsilon_upper: Upper bound offset (default 0.28 from DAPO)
        """
        self.epsilon_lower = epsilon_lower
        self.epsilon_upper = epsilon_upper
    
    def clip(self, probs_ratio: torch.Tensor) -> torch.Tensor:
        """
        Apply asymmetric clipping.
        
        Args:
            probs_ratio: (B, G, T) probability ratios
            
        Returns:
            Clipped ratios in range [1-ε_l, 1+ε_h]
        """
        return torch.clamp(
            probs_ratio,
            1 - self.epsilon_lower,
            1 + self.epsilon_upper
        )