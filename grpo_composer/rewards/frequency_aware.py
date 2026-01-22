"""
This Code Comes from the paper - "Group Aware Policy Optimisation - GAPO"
Group-Aware Policy Optimization (GAPO),
a simple extension of the recent and popular
Group Relative Policy Optimization (GRPO)
that computes rewards over the group as a
whole. GAPO enables learning from the group-
level properties such as diversity and cover-
age.

"""

"""

Implementation of the reward function from:
"Group-Aware Reinforcement Learning for Output Diversity in Large Language Models"
Amazon Research, 2025

Key Formula:
    f_v(o) = sum(1{o_i == v}) / sum(1{o_i in V})
    
    R(o)_i = 1 - (f_{o_i} - 1/L),  if o_i in V
           = -1,                    otherwise

Where:
    - V: Set of valid items
    - L: Number of valid items (|V|)
    - f_v: Frequency of item v among valid completions
    - G: Number of rollouts (group size)

Limitations:
    1. Assumes equally valid completions (NOT for math/factual QA)
    2. Only for tasks where multiple correct answers exist
    3. For list selection and open-ended creative tasks
"""

import torch
import numpy as np
from typing import List, Set, Dict, Union
from collections import Counter


class GAPOFrequencyReward:
    """
    Frequency-aware reward calculator for GAPO training.
    
    This reward function encourages uniform sampling over a set of 
    equally valid responses by penalizing over-represented items 
    and boosting under-represented ones.
    """
    
    def __init__(self, valid_set: Set[str], invalid_penalty: float = -1.0):
        """
        Initialize the reward calculator.
        
        Args:
            valid_set: Set of valid response items (V in the paper)
            invalid_penalty: Reward for invalid responses (default: -1.0)
        """
        self.valid_set = valid_set
        self.L = len(valid_set)  # Number of valid items
        self.u = 1.0 / self.L    # Target uniform probability
        self.invalid_penalty = invalid_penalty
        
        if self.L == 0:
            raise ValueError("valid_set cannot be empty")
    
    def compute_frequency(self, completions: List[str], item: str) -> float:
        """
        Compute empirical frequency of a specific item among valid completions.
        
        f_v(o) = sum(1{o_i == v}) / sum(1{o_i in V})
        
        Args:
            completions: List of G completions from the model
            item: Specific item to compute frequency for
        
        Returns:
            Frequency of the item (0.0 if no valid completions exist)
        """
        # Count how many times this item appears
        count_item = sum(1 for c in completions if c == item)
        
        # Count total valid completions
        count_valid = sum(1 for c in completions if c in self.valid_set)
        
        # Return frequency
        if count_valid == 0:
            return 0.0
        return count_item / count_valid
    
    def compute_reward(self, completion: str, completions: List[str]) -> float:
        """
        Compute frequency-aware reward for a single completion.
        
        R(o)_i = 1 - (f_{o_i} - 1/L),  if o_i in V
               = -1,                    otherwise
        
        Args:
            completion: The specific completion to compute reward for (o_i)
            completions: All G completions in the group (o)
        
        Returns:
            Scalar reward value
        """
        # Case 1: Invalid response
        if completion not in self.valid_set:
            return self.invalid_penalty
        
        # Case 2: Valid response
        # Compute frequency of this specific item
        f_oi = self.compute_frequency(completions, completion)
        
        # Reward formula: 1 - (f_oi - u)
        reward = 1.0 - (f_oi - self.u)
        
        return reward
    
    def compute_rewards(self, completions: List[str]) -> torch.Tensor:
        """
        Compute rewards for all completions in the group.
        
        This is the main function to use in GAPO training.
        
        Args:
            completions: List of G completions from the model
        
        Returns:
            Tensor of shape (G,) containing rewards for each completion
        """
        rewards = []
        
        for completion in completions:
            reward = self.compute_reward(completion, completions)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def get_statistics(self, completions: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Get detailed statistics about the completions and rewards.
        
        Useful for debugging and analysis.
        
        Args:
            completions: List of G completions
        
        Returns:
            Dictionary containing:
                - frequencies: Frequency of each valid item
                - rewards: Mean reward for each item
                - valid_ratio: Proportion of valid completions
                - entropy: Shannon entropy of the distribution
        """
        # Count occurrences
        valid_completions = [c for c in completions if c in self.valid_set]
        invalid_count = len(completions) - len(valid_completions)
        
        if len(valid_completions) == 0:
            return {
                "frequencies": {},
                "rewards": {},
                "valid_ratio": 0.0,
                "invalid_count": invalid_count,
                "entropy": 0.0
            }
        
        # Compute frequencies
        counter = Counter(valid_completions)
        frequencies = {item: count / len(valid_completions) 
                      for item, count in counter.items()}
        
        # Compute rewards for each item
        rewards_dict = {}
        for item in self.valid_set:
            if item in frequencies:
                f_item = frequencies[item]
                rewards_dict[item] = 1.0 - (f_item - self.u)
            else:
                # Item not sampled, hypothetical reward
                rewards_dict[item] = 1.0 - (0.0 - self.u)
        
        # Compute entropy (measure of uniformity)
        entropy = 0.0
        for freq in frequencies.values():
            if freq > 0:
                entropy -= freq * np.log2(freq)
        
        return {
            "frequencies": frequencies,
            "rewards": rewards_dict,
            "valid_ratio": len(valid_completions) / len(completions),
            "invalid_count": invalid_count,
            "entropy": entropy,
            "max_entropy": np.log2(self.L)  # Maximum possible entropy
        }
