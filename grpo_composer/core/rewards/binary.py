"""
Binary Reward Calculator (Base GRPO)

Implements simple binary/pass-through rewards for vanilla GRPO.
- Pass-through mode: Returns rewards unchanged
- Threshold mode: Converts continuous rewards to binary (0/1)

Input:  torch.Tensor of shape (batch_size, num_completions)
Output: torch.Tensor of shape (batch_size, num_completions)
"""

import torch
from typing import Optional
from .base import RewardCalculator


class BinaryRewardCalculator(RewardCalculator):
    """
    Binary reward calculator for base GRPO.
    
    Supports two modes:
    - Pass-through (threshold=None): Returns rewards unchanged
    - Thresholding (threshold=float): Converts to 0/1 based on threshold
    
    Args:
        rewards: torch.Tensor of shape (batch_size, num_completions)
        threshold: Optional threshold value. If None, pass-through mode.
                   If float, values > threshold become 1, else 0.
    """
    
    def __init__(
        self, 
        rewards: torch.Tensor, 
        threshold: Optional[float] = None, 
        **kwargs
    ) -> None:
        super().__init__(rewards, **kwargs)
        self.threshold = threshold
        
        # Validate input shape
        if rewards.ndim != 2:
            raise ValueError(f"Rewards must be 2D tensor, got shape {rewards.shape}")

    def compute_rewards(self) -> torch.Tensor:
        """
        Compute binary rewards.
        
        Returns:
            torch.Tensor of shape (batch_size, num_completions)
            - If threshold is None: unchanged rewards
            - If threshold is float: binary 0/1 rewards
        """
        if self.threshold is not None:
            return torch.where(
                self.rewards > self.threshold,
                torch.ones_like(self.rewards),
                torch.zeros_like(self.rewards)
            )
        return self.rewards
