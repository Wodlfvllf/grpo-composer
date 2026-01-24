"""
Posterior Composite Reward Calculator (P-GRPO)

Implements the composite reward from "Posterior-GRPO: Rewarding Reasoning
Processes in Code Generation".

Formula:
    R_i = R_f + R_o + R_o * R_t

Where:
    - R_f ∈ {0, 1}: Format reward (structural compliance)
    - R_o ∈ {0, 1}: Outcome reward (test case verification)  
    - R_t ∈ [0, 1]: Thinking reward from reward model

The posterior gating (R_o * R_t) ensures thinking reward only
contributes when the outcome is correct.

Input:
    responses: List of response strings
    format_fn: Callable that returns format reward (0 or 1)
    outcome_fn: Callable that returns outcome reward (0 or 1)
    thinking_fn: Callable that returns thinking reward [0, 1]
    
Output:
    torch.Tensor of shape (num_completions,)
"""

import torch
from typing import List, Callable, Optional, Union
from .base import RewardCalculator


class PosteriorCompositeRewardCalculator(RewardCalculator):
    """
    P-GRPO: Posterior-based composite reward with posterior gating.
    
    Combines format, outcome, and thinking rewards with the thinking
    reward gated by outcome correctness.
    
    Args:
        responses: List of response strings
        format_fn: Callable(response) -> float (0 or 1)
        outcome_fn: Callable(response) -> float (0 or 1)
        thinking_fn: Callable(response) -> float [0, 1]
        
    Alternative constructor with pre-computed rewards:
        Use from_precomputed() class method
    """
    
    def __init__(
        self,
        responses: List[str],
        format_fn: Callable[[str], float],
        outcome_fn: Callable[[str], float],
        thinking_fn: Callable[[str], float],
        **kwargs
    ) -> None:
        dummy_rewards = torch.zeros(len(responses))
        super().__init__(dummy_rewards, **kwargs)
        
        self.responses = responses
        self.format_fn = format_fn
        self.outcome_fn = outcome_fn
        self.thinking_fn = thinking_fn
        
        # Pre-computed rewards (set by from_precomputed)
        self._format_rewards: Optional[torch.Tensor] = None
        self._outcome_rewards: Optional[torch.Tensor] = None
        self._thinking_rewards: Optional[torch.Tensor] = None
    
    @classmethod
    def from_precomputed(
        cls,
        format_rewards: torch.Tensor,
        outcome_rewards: torch.Tensor,
        thinking_rewards: torch.Tensor,
    ) -> "PosteriorCompositeRewardCalculator":
        """
        Create calculator from pre-computed reward tensors.
        
        Args:
            format_rewards: shape (num_completions,) binary 0/1
            outcome_rewards: shape (num_completions,) binary 0/1
            thinking_rewards: shape (num_completions,) float [0, 1]
            
        Returns:
            PosteriorCompositeRewardCalculator instance
        """
        n = len(format_rewards)
        if len(outcome_rewards) != n or len(thinking_rewards) != n:
            raise ValueError("All reward tensors must have same length")
        
        # Create instance with dummy functions
        instance = cls(
            responses=[""] * n,
            format_fn=lambda x: 0.0,
            outcome_fn=lambda x: 0.0,
            thinking_fn=lambda x: 0.0,
        )
        
        # Set pre-computed rewards
        instance._format_rewards = format_rewards
        instance._outcome_rewards = outcome_rewards
        instance._thinking_rewards = thinking_rewards
        
        return instance

    def compute_rewards(self) -> torch.Tensor:
        """
        Compute composite rewards using P-GRPO formula.
        
        R_i = R_f + R_o + R_o * R_t
        
        Returns:
            torch.Tensor of shape (num_completions,)
        """
        # Use pre-computed if available
        if self._format_rewards is not None:
            r_f = self._format_rewards
            r_o = self._outcome_rewards
            r_t = self._thinking_rewards
        else:
            # Compute from functions
            r_f = torch.tensor([self.format_fn(r) for r in self.responses])
            r_o = torch.tensor([self.outcome_fn(r) for r in self.responses])
            r_t = torch.tensor([self.thinking_fn(r) for r in self.responses])
        
        # P-GRPO formula: R = R_f + R_o + R_o * R_t
        return r_f + r_o + r_o * r_t
