
"""
Standard GRPO Loss Implementation

This module implements the standard GRPO loss composition.

Role:
-----
- Combines the policy gradient term (computed by AggregationFunction) with KL regularization.
- Converts the Maximization problem (Max R(theta)) into a Minimization problem (Min Loss(theta)).

Formula:
--------
Loss = - ( Aggregation(adv, ratio) - a * KL_Penalty(pi, pi_ref) )

Where:
- Aggregation(adv, ratio) comes from `grpo_composer.aggregation`
- KL_Penalty(pi, pi_ref) comes from `grpo_composer.kl_penalties` (future)

"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base import LossFunction
from ..aggregation import AggregationFunction

class CustomLoss(LossFunction):
    """
    Standard GRPO Loss:
    
    L = - ( L_policy - β * D_KL(π || π_ref) )
    
    Args:
        aggregation_fn (AggregationFunction): The specific implementation of the policy surrogate 
            (e.g. TokenMean, GlobalToken, Dr.GRPO/TokenSum).
        kl_penalty_fn (Optional[Any]): The implementation of KL penalty 
            (e.g. standard KL, reverse KL). To be implemented in `grpo_composer.kl_penalties`.
        kl_weight (float): The coefficient β for the KL term. Default is 0.1 (can be adaptive).
    """
    
    def __init__(
        self, 
        aggregation_fn: AggregationFunction, 
        kl_penalty_fn: Any = None, # Will be KLPenaltyFunction
        kl_weight: float = 0.1
    ):
        super().__init__()
        self.aggregation_fn = aggregation_fn
        self.kl_penalty_fn = kl_penalty_fn
        self.kl_weight = kl_weight
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        mask: torch.Tensor,             # (B, G, T)
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes - ( Policy_Surrogate - β * KL )
        
        Note regarding KL:
        ------------------
        In many GRPO implementations (DeepSeek/HuggingFace), the KL is computed per-token 
        and subtracted from the REWARD before advantage calculation.
        
        However, in this modular design:
        - We support BOTH methods.
        - If KL is subtracted from Rewards ("Reward-KL"): Set kl_weight=0 or pass None for kl_penalty_fn.
        - If KL is a separate regularizer ("Constraint-KL"): Pass kl_penalty_fn here.
        
        Standard GRPO (DeepSeekMath) typically uses "Reward-KL".
        """
        
        # 1. Compute Policy Surrogate Loss (L_policy)
        # Delegation: The aggregation function handles advantages, ratios, weighting, clipping.
        # It returns a SCALAR or average per-token value representing "goodness".
        policy_surrogate = self.aggregation_fn.compute_aggregation(
            rewards, log_probs, ref_log_probs, mask
        )
        
        # 2. Compute KL Penalty (L_kl) - OPTIONAL here if done in rewards
        kl_term = torch.tensor(0.0, device=log_probs.device)
        if self.kl_penalty_fn is not None:
            # Future: kl_term = self.kl_penalty_fn(log_probs, ref_log_probs, mask).mean()
            pass 

        # 3. Combine: minimize negative objective
        # Objective = Policy - beta * KL
        # Loss = -Objective = -Policy + beta * KL
        total_loss = -policy_surrogate + (self.kl_weight * kl_term)
        
        metrics = {
            "policy_surrogate": policy_surrogate.item(),
            "kl_penalty": kl_term.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, metrics
