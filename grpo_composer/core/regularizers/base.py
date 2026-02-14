"""
Base Regularizer Module

This module defines the abstract base class for all regularization terms in the GRPO library.

Regularizers are additive terms to the loss function that encourage specific properties
(e.g., closeness to reference model, preference alignment, curriculum balancing).

Interface:
    compute_regularization(model_outputs, ref_outputs, inputs) -> (loss_term, metrics)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

class Regularizer(nn.Module):
    """
    Abstract base class for GRPO regularizers.
    
    A regularizer computes a scalar penalty/bonus to be added to the total loss.
    L_total = L_surrogate + coefficient * L_regularizer
    
    Subclasses must implement:
        compute_regularization(...)
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        rewards: torch.Tensor,          # (B, G)
        mask: torch.Tensor,             # (B, G, T)
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the regularization term.
        
        Args:
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference model
            rewards: Rewards for each generation
            mask: Attention mask for valid tokens
            **kwargs: Additional contextual data (e.g., latent variables z for Info-GRPO)
            
        Returns:
            reg_loss: Scalar tensor representing the regularization term (to be minimized)
        """
        raise NotImplementedError