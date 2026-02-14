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


class LengthDependentRewardCalculator:
    """
    GRPO-LEAD: Length-Dependent Accuracy Reward.
    
    Promotes brevity among correct responses using standardized
    length-based penalties.
    
    Args:
        alpha: Length penalty strength (default: 0.05 from paper)
        epsilon: Numerical stability constant (default: 1e-8)
        incorrect_penalty: Reward for incorrect responses (default: -1.0)
    
    Example:
        >>> calc = LengthDependentRewardCalculator(alpha=0.05)
        >>> responses = [[1,2,3], [1,2,3,4,5,6], [1,2]]
        >>> labels = torch.tensor([1, 1, 0])
        >>> rewards = calc.compute_rewards(responses, labels)
        # Short correct boosted, long correct penalized, wrong = -1
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        epsilon: float = 1e-8,
        incorrect_penalty: float = -1.0
    ) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.incorrect_penalty = incorrect_penalty

    def _get_length(self, response: List[int]) -> int:
        """Get token length of a response."""
        return len(response)
    
    def _compute_length_stats(
        self, 
        responses: List[List[int]], 
        labels: torch.Tensor
    ) -> tuple:
        """
        Compute mean and std of CORRECT response lengths.
        
        Args:
            responses: List of token sequences
            labels: (N,) binary correctness labels
            
        Returns:
            (mean_length, std_length)
        """
        correct_lengths = [
            self._get_length(resp) 
            for resp, label in zip(responses, labels.tolist()) 
            if label == 1
        ]
        if len(correct_lengths) == 0:
            return 0.0, 1.0  # Fallback if no correct responses
            
        mean_length = np.mean(correct_lengths)
        std_length = np.std(correct_lengths)
        return mean_length, std_length

    def compute_rewards(
        self, 
        responses: List[List[int]], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute length-dependent rewards.
        
        Args:
            responses: List of token sequences (each is List[int])
            labels: (N,) or (B, G) binary correctness labels (1=correct, 0=incorrect)
        
        Returns:
            torch.Tensor of same shape as labels
        """
        # Handle batched input (B, G)
        if labels.dim() == 2:
            B, G = labels.shape
            flat_labels = labels.view(-1)
            # Flatten responses if nested
            if isinstance(responses[0][0], list):
                flat_responses = [r for batch in responses for r in batch]
            else:
                flat_responses = responses
            flat_rewards = self._compute_single(flat_responses, flat_labels)
            return flat_rewards.view(B, G)
        
        return self._compute_single(responses, labels)
    
    def _compute_single(
        self, 
        responses: List[List[int]], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards for flat list of responses."""
        mean_length, std_length = self._compute_length_stats(responses, labels)
        
        rewards = []
        for response, label in zip(responses, labels.tolist()):
            if label == 1:
                length = self._get_length(response)
                z = (length - mean_length) / (std_length + self.epsilon)
                reward = float(np.exp(-self.alpha * z))
            else:
                reward = self.incorrect_penalty
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)