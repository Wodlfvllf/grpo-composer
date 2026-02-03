
"""
This is the class that implements the standard advantage function.
"""

import torch
from .base import AdvantageFunction

class StandardAdvantageFunction(AdvantageFunction):
    """
    Standard advantage function.
    """
    def __init__(self):
        super().__init__()
    
    def compute_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages.
        
        Args:
            rewards: (B, G) rewards
            
        Returns:
            advantages: (B, G) advantages
        """
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        return (rewards - mean) / (std + 1e-8)