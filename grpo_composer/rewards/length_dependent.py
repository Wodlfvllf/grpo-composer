"""
Length-Dependent Reward Calculator (GRPO-LEAD)

Implements length-dependent accuracy reward that encourages concise
mathematical reasoning by penalizing verbose solutions.

Formula:
    z_i = (length(o_i) - μ) / (σ + ε)
    R_accuracy(o|q) = exp(-α * z_i),  if o is correct
                    = -1,              if o is incorrect

Where:
    - μ: Mean length of CORRECT responses only
    - σ: Standard deviation of CORRECT response lengths
    - z_i: Standardized length deviation
    - α: Length penalty strength (paper uses α=0.05)
    - ε: Numerical stability constant

Design:
    - Statistics computed ONLY over correct responses
    - Incorrect responses get fixed penalty of -1
    - Shorter correct responses get reward > 1.0 (boost)
    - Longer correct responses get reward < 1.0 (penalty)

Input:  
    responses: List of token sequences (List[List[int]])
    labels: Binary correctness labels (1=correct, 0=incorrect)
Output: 
    torch.Tensor of shape (num_completions,)
"""

import torch
import numpy as np
from typing import List, Union
from .base import RewardCalculator


class LengthDependentRewardCalculator(RewardCalculator):
    """
    GRPO-LEAD: Length-Dependent Accuracy Reward.
    
    Promotes brevity among correct responses using standardized
    length-based penalties.
    
    Args:
        responses: List of token sequences (each is List[int])
        labels: Binary correctness labels (1=correct, 0=incorrect)
        alpha: Length penalty strength (default: 0.05 from paper)
        epsilon: Numerical stability constant (default: 1e-8)
    
    Example:
        >>> responses = [[1,2,3], [1,2,3,4,5,6], [1,2]]
        >>> labels = [1, 1, 0]
        >>> calc = LengthDependentRewardCalculator(responses, labels)
        >>> rewards = calc.compute_rewards()
        # Short correct boosted, long correct penalized, wrong = -1
    """
    
    def __init__(
        self,
        responses: List[List[int]],
        labels: List[int],
        alpha: float = 0.05,
        epsilon: float = 1e-8,
        incorrect_penalty: float = -1.0,
        **kwargs
    ) -> None:
        # Create dummy rewards for base class
        dummy_rewards = torch.zeros(len(responses))
        super().__init__(dummy_rewards, **kwargs)
        
        if len(responses) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(responses)} responses but {len(labels)} labels"
            )
        
        self.responses = responses
        self.labels = labels
        self.alpha = alpha
        self.epsilon = epsilon
        self.incorrect_penalty = incorrect_penalty
        
        # Validate we have at least one correct response
        if sum(labels) == 0:
            raise ValueError(
                "No correct responses found. GRPO-LEAD requires at least "
                "one correct response to compute length statistics."
            )

    def _get_length(self, response: List[int]) -> int:
        """Get token length of a response."""
        if not isinstance(response, list):
            raise TypeError(f"Response must be list of tokens, got {type(response)}")
        return len(response)
    
    def _compute_length_stats(self) -> tuple:
        """
        Compute mean and std of CORRECT response lengths.
        
        Returns:
            (mean_length, std_length)
        """
        correct_lengths = [
            self._get_length(resp) 
            for resp, label in zip(self.responses, self.labels) 
            if label == 1
        ]
        mean_length = np.mean(correct_lengths)
        std_length = np.std(correct_lengths)
        return mean_length, std_length

    def compute_rewards(self) -> torch.Tensor:
        """
        Compute length-dependent rewards.
        
        Returns:
            torch.Tensor of shape (num_completions,)
        """
        mean_length, std_length = self._compute_length_stats()
        
        rewards = []
        for response, label in zip(self.responses, self.labels):
            if label == 1:
                length = self._get_length(response)
                z = (length - mean_length) / (std_length + self.epsilon)
                reward = float(np.exp(-self.alpha * z))
            else:
                reward = self.incorrect_penalty
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)