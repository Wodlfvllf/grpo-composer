"""
Frequency-Aware Reward Calculator (GAPO)

Group-Aware Policy Optimization: computes rewards based on answer frequency
to encourage diversity in valid responses.

Formula:
    f_v(o) = count(o_i == v) / count(o_i in V)
    R(o)_i = 1 - |f_{o_i} - 1/L|,  if o_i in V
           = -1,                    otherwise

Where:
    - V: Set of valid items
    - L: |V| (number of valid items)
    - f_v: Frequency of item v among valid completions

Note: This reward is for tasks with multiple equally-valid answers (e.g., 
list selection, creative tasks), NOT for math/factual QA with single answers.

Input:  List of string completions (single group)
Output: torch.Tensor of shape (num_completions,)
"""

import torch
import numpy as np
from typing import List, Set, Dict, Union, Optional
from collections import Counter
from .base import RewardCalculator


class FrequencyAwareRewardCalculator(RewardCalculator):
    """
    GAPO: Frequency-aware reward calculator that encourages uniform 
    sampling over valid responses.
    
    Args:
        completions: List of G string completions from the model
        valid_set: Set of valid response items (V in the paper)
        invalid_penalty: Reward for invalid responses (default: -1.0)
    """
    
    def __init__(
        self,
        completions: List[str],
        valid_set: Set[str],
        invalid_penalty: float = -1.0,
        **kwargs
    ) -> None:
        # Create dummy rewards tensor for base class
        dummy_rewards = torch.zeros(len(completions))
        super().__init__(dummy_rewards, **kwargs)
        
        self.completions = completions
        self.valid_set = valid_set
        self.invalid_penalty = invalid_penalty
        self.L = len(valid_set)
        self.u = 1.0 / self.L if self.L > 0 else 0.0
        
        if self.L == 0:
            raise ValueError("valid_set cannot be empty")
    
    def _compute_frequency(self, item: str) -> float:
        """
        Compute empirical frequency of item among valid completions.
        
        f_v(o) = count(o_i == v) / count(o_i in V)
        """
        count_item = sum(1 for c in self.completions if c == item)
        count_valid = sum(1 for c in self.completions if c in self.valid_set)
        
        if count_valid == 0:
            return 0.0
        return count_item / count_valid
    
    def _compute_single_reward(self, completion: str) -> float:
        """
        Compute frequency-aware reward for a single completion.
        
        R(o)_i = 1 - |f_{o_i} - 1/L|,  if o_i in V
               = -1,                    otherwise
        """
        if completion not in self.valid_set:
            return self.invalid_penalty
        
        f_oi = self._compute_frequency(completion)
        return 1.0 - abs(f_oi - self.u)
    
    def compute_rewards(self) -> torch.Tensor:
        """
        Compute rewards for all completions.
        
        Returns:
            torch.Tensor of shape (num_completions,)
        """
        rewards = [self._compute_single_reward(c) for c in self.completions]
        return torch.tensor(rewards, dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get detailed statistics for debugging and analysis.
        
        Returns:
            Dictionary with frequencies, rewards, valid_ratio, entropy
        """
        valid_completions = [c for c in self.completions if c in self.valid_set]
        invalid_count = len(self.completions) - len(valid_completions)
        
        if len(valid_completions) == 0:
            return {
                "frequencies": {},
                "rewards": {},
                "valid_ratio": 0.0,
                "invalid_count": invalid_count,
                "entropy": 0.0
            }
        
        counter = Counter(valid_completions)
        frequencies = {
            item: count / len(valid_completions) 
            for item, count in counter.items()
        }
        
        rewards_dict = {}
        for item in self.valid_set:
            f_item = frequencies.get(item, 0.0)
            rewards_dict[item] = 1.0 - abs(f_item - self.u)
        
        entropy = -sum(
            f * np.log2(f) for f in frequencies.values() if f > 0
        )
        
        return {
            "frequencies": frequencies,
            "rewards": rewards_dict,
            "valid_ratio": len(valid_completions) / len(self.completions),
            "invalid_count": invalid_count,
            "entropy": entropy,
            "max_entropy": np.log2(self.L)
        }
