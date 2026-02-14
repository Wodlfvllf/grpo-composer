"""
Log-Weight Regularizer

This module implements the regularization term for DARO (Difficulty-Aware Regularization Optimization).

Paper: DARO
-----------
Adds a log-weight regularizer to prevent the learnable difficulty weights w_μ from collapsing
or exploding, encouraging a balanced curriculum.

Formula:
--------
L_reg = - Σ_{μ} ln(w_μ)

Where:
- w_μ are the learnable weights for each difficulty bin μ.
- Weights are constrained to be positive (w_μ > 0).

Effect:
-------
- Penalizes small weights strongly (simulating a barrier function).
- Encourages weights to be non-zero.
- Combined with the main DARO loss term (w_μ * L_μ), this leads to the optimal solution
  where w*_μ is inversely proportional to the group loss L_μ.
"""

import torch
from typing import Any
from .base import Regularizer


class LogWeightRegularizer(Regularizer):
    """
    DARO Log-Weight Regularizer.
    
    Regularizes the learnable weights w_μ associated with difficulty bins.
    Typically used in conjunction with DifficultyWeightedAggregation.
    """
    
    def __init__(self, num_bins: int = 10, eps: float = 1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps
        
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, G, T) - Unused, required by base interface
        ref_log_probs: torch.Tensor,    # (B, G, T) - Unused, required by base interface
        rewards: torch.Tensor,          # (B, G) - Unused, required by base interface
        mask: torch.Tensor,             # (B, G, T) - Unused, required by base interface
        weights: torch.Tensor = None,   # (num_bins,) - The learnable weights w_μ
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the log-weight regularization term.
        
        Args:
            weights: Tensor containing the learnable weights w_μ.
            
        Returns:
            reg_loss: - sum(log(weights))
            
        Raises:
            ValueError: If weights is None or has incorrect shape.
        """
        if weights is None:
            raise ValueError("LogWeightRegularizer requires 'weights' to be provided.")
        
        if weights.shape[0] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} weights, got {weights.shape[0]}")
        
        # Numerical stability: clamp weights to avoid log(0)
        reg_loss = -torch.sum(torch.log(weights.clamp(min=self.eps)))

        return reg_loss

        
