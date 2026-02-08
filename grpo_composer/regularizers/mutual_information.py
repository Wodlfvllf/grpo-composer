"""
Mutual Information Regularizer (Info-GRPO)

This module implements the MI-based regularization term for Info-GRPO.

Paper: Info-GRPO
----------------
Addresses the gradient instability of standard entropy used in PPO/GRPO when token probabilities
are extremely small. Instead of maximizing entropy H(π), it maximizes the mutual information
I(τ, z) between trajectories τ and latent variables z.

Formula:
--------
I(τ, z) = H(π) - H(π|z)

Info Regularizer:
    J_Info(θ) = α * H(π_θ(τ|s0)) - H(π_θ(τ|s0, z))

Implementation Logic:
---------------------
1. Standard Entropy H(π): computed over original trajectories (T_ori).
2. Conditional Entropy H(π|z): computed over augmented trajectories (T_aug) conditioned on z.

Loss Term (to minimize):
    L_reg = - λ * ( α * H(T_ori) - H(T_aug|z) )

Wait, the paper says:
    J_Info-GRPO = J_GRPO(T_ori ∪ T_aug) + λ * ( α * H(T_ori) - H(T_aug|z) )

For the REGULARIZER component specifically, we compute the second term.
Input requires two sets of log-probs: one from normal generation (T_ori), 
one from z-conditioned generation (T_aug).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .base import Regularizer

class MutualInformationRegularizer(Regularizer):
    """
    Info-GRPO Mutual Information Regularizer.
    
    Maximizes mutual information I(τ, z) = H(π) - H(π|z) to stabilize training.
    Avoiding singularities of standard entropy at low probabilities.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,  # Weight for unconditional entropy
    ):
        super().__init__()
        self.alpha = alpha
        
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        mask: torch.Tensor,             # (B, G, T)
        log_probs_aug: torch.Tensor = None, # (B, G, T) - From z-conditioned generation
        mask_aug: torch.Tensor = None,      # (B, G, T) - Mask for augmented generation
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the MI regularization term.
        
        Args:
            log_probs: Log-probs from standard generation (T_ori)
            log_probs_aug: Log-probs from latent-conditioned generation (T_aug|z). Required.
            mask_aug: Mask for augmented rollouts.
            
        Returns:
            reg_loss: - ( alpha * H(T_ori) - H(T_aug|z) )
            
        Note:
            H(T) ≈ - mean( Σ log_probs )
            So maximizing H means minimizing -H.
            
            We want to MAXIMIZE (alpha * H_ori - H_aug).
            Equivalent to MINIMIZING - (alpha * H_ori - H_aug) = -alpha * H_ori + H_aug.
        """
        if log_probs_aug is None:
            raise ValueError("MutualInformationRegularizer requires 'log_probs_aug'.")
        if mask_aug is None:
            raise ValueError("MutualInformationRegularizer requires 'mask_aug'.")
            
        # ---- Unconditional entropy (control term, detached) ----
        token_count = mask.sum(dim=-1) + 1e-8
        H_ori = - (log_probs * mask).sum(dim=-1) / token_count
        H_ori = H_ori.mean().detach()   # stop-gradient

        # ---- Conditional entropy (dominant term) ----
        token_count_aug = mask_aug.sum(dim=-1) + 1e-8
        H_aug = - (log_probs_aug * mask_aug).sum(dim=-1) / token_count_aug
        H_aug = H_aug.mean()

        # ---- MI regularizer loss ----
        reg_loss = - (self.alpha * H_ori - H_aug)

        return reg_loss
