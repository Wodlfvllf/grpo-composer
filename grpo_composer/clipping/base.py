
"""
Base Abstract Class For Clipping in GRPO Mechanisms

Clipping mechanisms control how much the policy can change in a single update.
Different variants modify the clipping strategy:
- Symmetric: Standard PPO-style [1-ε, 1+ε] bounds
- Asymmetric: DAPO/P-GRPO style [1-ε_l, 1+ε_h] with different bounds
- Trajectory-level: TIC-GRPO clips at sequence level, not token level
- Weighted trust: TR-GRPO uses confidence-weighted dynamic bounds
"""

import torch
from abc import ABC, abstractmethod
from typing import Union


class ClippingMechanism(ABC):
    """
    Abstract base class for all clipping mechanisms.
    
    All clipping mechanisms receive probability ratios:
        ρ = π_θ(a|s) / π_θ_old(a|s)
    
    And return clipped ratios that constrain how much the policy can change.
    """
    
    @abstractmethod
    def clip(self, probs_ratio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply clipping to probability ratios.
        
        Args:
            probs_ratio: Probability ratios, typically shape (B, G, T) for token-level
                        or (B, G) for trajectory-level
            **kwargs: Additional arguments specific to clipping variant
            
        Returns:
            Clipped ratios with same shape as input
        """
        pass
