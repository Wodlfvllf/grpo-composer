
"""
Base Abstract Class For Advantages in GRPO Mechanisms

"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union


class AdvantageFunction(ABC):
    """
    Abstract base class for advantage functions.
    """
    
    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages.
        
        Args:
            rewards: (B, G, T) rewards
            
        Returns:
            advantages: (B, G, T) advantages
        """
        pass