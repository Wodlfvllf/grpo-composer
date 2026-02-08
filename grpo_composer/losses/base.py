
"""
Base Loss Function Module

This module defines the abstract base class for all loss functions in the GRPO library.

Role of this module (Losses) vs Aggregation:
--------------------------------------------
1. Aggregation (grpo_composer.aggregation):
   - Focuses ONLY on the Policy Gradient Surrogate.
   - Computes: E[ ratio * advantage ] using various weighting schemes (Token-Mean, Global-Sum, Difficulty-Weighted, etc.).
   - Returns a scalar or per-token tensor representing the "goodness" of the policy update.
   - Does NOT handle KL divergence or entropy regularization.

2. Losses (grpo_composer.losses):
   - Orchestrates the final optimization objective.
   - Combines the Aggregation result with Regularization terms.
   - Typical Formula: Loss = - ( Policy_Surrogate - beta * KL_Divergence + entropy_bonus )
   - Handles the conversion from Maximization (RL objective) to Minimization (PyTorch optimizer).

What goes in here?
------------------
- Base class defining the `compute_loss` interface.
- Handling of auxiliary outputs (e.g., logging metrics like mean_kl, mean_entropy).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

class LossFunction(nn.Module):
    """
    Abstract base class for GRPO loss functions.
    
    The LossFunction is the final consumer of:
    1. Model outputs (log_probs)
    2. Reference model outputs (ref_log_probs) for KL
    3. Rewards/Advantages
    
    It delegates:
    - Surrogate computation -> AggregationFunction
    - KL computation -> KLPenalty (future component)
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        mask: torch.Tensor,             # (B, G, T)
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the final scalar loss and metrics.
        
        Args:
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference model
            rewards: Rewards for each generation
            mask: Attention mask for valid tokens
            **kwargs: Additional inputs (e.g., values for value-based methods)
            
        Returns:
            loss: Scalar tensor for backpropagation (minimized)
            metrics: Dictionary of metrics for logging (policy_loss, kl, entropy, etc.)
        """
        raise NotImplementedError
