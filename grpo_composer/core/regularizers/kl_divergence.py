"""
KL Divergence Regularizers

This module implements Kullback-Leibler (KL) divergence penalties.
KL regularization keeps the policy close to the reference model to prevent reward hacking
and maintain generation quality.

Paper References:
-----------------
1. Standard KL (PPO/GRPO defaults):
   - Used in initial PPO, effectively beta * KL.
   
2. No-KL Variants (beta=0):
   - Dr.GRPO: Removes KL penalty entirely.
   - DAPO: Removes KL penalty entirely.
   - GRPO-LEAD: Removes KL penalty (found to suppress exploration).
   - P-GRPO: Removes KL penalty.

3. TR-GRPO: Weighted KL Divergence
   - D_KL = sum( w_{i,t} * KL(pi_theta || pi_ref) )
   - Token weight modulates KL contribution per token.

Formulas:
---------
Standard Forward KL:
    D_KL(π || π_ref) = Σ π(x) log( π(x) / π_ref(x) )
    Appximated as: log_probs - ref_log_probs (since expectations are over samples from π)

Weighted KL (TR-GRPO):
    D_KL_weighted = Σ w_{i,t} * (log π_θ(o_t|...) - log π_ref(o_t|...))
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .base import Regularizer

class KLDivergenceRegularizer(Regularizer):
    """
    Standard KL Divergence Regularizer.
    
    Computes the KL divergence between the current policy and the reference model.
    Used for standard PPO/GRPO when beta > 0.
    
    For methods that remove KL (Dr.GRPO, DAPO, GRPO-LEAD, P-GRPO),
    simply set the weight coefficient (beta) to 0 in the Loss configuration,
    or do not include this regularizer.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        mask: torch.Tensor,             # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes standard token-level KL divergence.
        
        Formula:
            KL ≈ log_probs - ref_log_probs (reverse KL: P log(P/Q))
            Calculated as mean over valid tokens.
        """
        # KL per token: (B, G, T)
        # Note: log(P/Q) = log(P) - log(Q)
        kl_tokens = (log_probs - ref_log_probs) * mask
        
        # Mean over valid tokens only (avoid diluting with padding)
        kl_loss = kl_tokens.sum() / (mask.sum() + 1e-8)
        
        return kl_loss

class WeightedKLDivergenceRegularizer(Regularizer):
    """
    Weighted KL Divergence Regularizer (TR-GRPO).
    
    Paper: TR-GRPO
    
    Formula:
        D_KL_weighted = Σ w_{i,t} * D_KL(π_θ || π_ref)
    
    Token weights w_{i,t} modulate the KL contribution per token, 
    allowing stricter constraints on some tokens and looser on others.
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        mask: torch.Tensor,             # (B, G, T)
        weights: torch.Tensor = None,   # (B, G, T) - Required for TR-GRPO
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes weighted KL divergence.
        
        Args:
            weights: Tensor of token weights w_{i,t}. Must be provided.
        """
        if weights is None:
            raise ValueError("WeightedKLDivergenceRegularizer requires 'weights' argument.")
            
        # KL per token: (B, G, T)
        kl = log_probs - ref_log_probs

        # Weighted KL: (B, G, T)
        weighted_kl = kl * weights * mask

        # Mean over valid tokens (normalized by total token count or total weight?)
        # Usually regularizers are added to loss which is mean-reduced.
        # We normalize by total valid tokens to keep scale consistent with policy loss.
        kl_loss = weighted_kl.sum() / (mask.sum() + 1e-8)
        
        return kl_loss
