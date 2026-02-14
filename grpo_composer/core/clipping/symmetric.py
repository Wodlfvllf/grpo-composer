
"""
Symmetric Clipping Mechanism for GRPO (Base GRPO / DeepSeekMath)

Standard PPO-style clipping with equal bounds on both sides:
    clip(ρ, 1-ε, 1+ε)

Default ε = 0.2 (standard PPO/GRPO value)

Input:
------
- probs_ratio: torch.Tensor, shape (B, G, T) - token-level probability ratios

Output:
-------
- clipped_ratio: torch.Tensor, shape (B, G, T)
"""

import torch
from .base import ClippingMechanism


class SymmetricClippingMechanism(ClippingMechanism):
    """
    Standard symmetric clipping from PPO/Base GRPO.
    
    clip(ρ, 1-ε, 1+ε) where ε is the same on both sides.
    """
    
    def __init__(self, epsilon: float = 0.2):
        """
        Args:
            epsilon: Clipping bound (default 0.2 from PPO/GRPO)
        """
        self.epsilon = epsilon
    
    def clip(self, probs_ratio: torch.Tensor) -> torch.Tensor:
        """
        Apply symmetric clipping.
        
        Args:
            probs_ratio: (B, G, T) probability ratios
            
        Returns:
            Clipped ratios in range [1-ε, 1+ε]
        """
        return torch.clamp(probs_ratio, 1 - self.epsilon, 1 + self.epsilon)