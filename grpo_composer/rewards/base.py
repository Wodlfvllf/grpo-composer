"""
Base Reward Calculator for GRPO Composer.

All reward calculators should inherit from this base class to ensure
a consistent API across the library.

Expected Input: torch.Tensor of shape (batch_size, num_completions)
Expected Output: torch.Tensor of shape (batch_size, num_completions)
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class RewardCalculator(ABC):
    """
    Abstract base class for all reward calculators in grpo-composer.
    
    Attributes:
        rewards: Input rewards tensor of shape (batch_size, num_completions)
    
    Subclasses must implement the `compute_rewards()` method.
    """
    
    def __init__(self, rewards: torch.Tensor, **kwargs: Any) -> None:
        """
        Initialize the reward calculator.
        
        Args:
            rewards: torch.Tensor of shape (batch_size, num_completions)
            **kwargs: Additional arguments stored as attributes
        """
        self.rewards = rewards
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def compute_rewards(self) -> torch.Tensor:
        """
        Compute and return transformed rewards.
        
        Returns:
            torch.Tensor of shape (batch_size, num_completions)
        """
        pass
    
    # Alias for backwards compatibility
    def calculate(self) -> torch.Tensor:
        """Alias for compute_rewards() for backwards compatibility."""
        return self.compute_rewards()
