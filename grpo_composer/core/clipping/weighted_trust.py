
"""
Weighted Trust Region Clipping Mechanism (TR-GRPO)

From: TR-GRPO paper

Mathematical Form:
------------------
Token weight:
    w_{i,t} = clip(α · (σ(π_θ(o_{i,t}) / τ) - μ), L, U)

Where:
- π_θ(o_{i,t}): Token probability under current policy
- τ: Temperature for scaling
- σ: Sigmoid function
- μ: Offset (typically 0.5 to center sigmoid output)
- α: Scaling factor
- L, U: Clipping bounds for weights

The weighted ratio becomes:
    w_{i,t} · r_{i,t}(θ) instead of just r_{i,t}(θ)

This allows dynamic trust region bounds per token based on the
model's confidence, controlling sharpness of the policy.

Input:
------
- probs_ratio: torch.Tensor, shape (B, G, T) - token-level probability ratios
- token_probs: torch.Tensor, shape (B, G, T) - token probabilities under current policy

Output:
-------
- weighted_clipped_ratio: torch.Tensor, shape (B, G, T)
"""

import torch
from .base import ClippingMechanism


class WeightedTrustRegionClippingMechanism(ClippingMechanism):
    """
    TR-GRPO: Probability-weighted trust region for sharpness control.
    
    Dynamically adjusts trust region bounds per token based on
    the model's confidence (token probability).
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        tau: float = 1.0,
        mu: float = 0.5,
        weight_lower: float = 0.5,
        weight_upper: float = 1.5,
        clip_epsilon: float = 0.2
    ):
        """
        Args:
            alpha: Scaling factor for weight computation
            tau: Temperature for sigmoid scaling
            mu: Offset for centering sigmoid output
            weight_lower: Lower bound for token weights (L)
            weight_upper: Upper bound for token weights (U)
            clip_epsilon: Base epsilon for ratio clipping
        """
        self.alpha = alpha
        self.tau = tau
        self.mu = mu
        self.weight_lower = weight_lower
        self.weight_upper = weight_upper
        self.clip_epsilon = clip_epsilon
    
    def compute_token_weights(self, token_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token weights based on model confidence.
        
        w_{i,t} = clip(α · (σ(π_θ(o_{i,t}) / τ) - μ), L, U)
        
        Args:
            token_probs: (B, G, T) token probabilities
            
        Returns:
            weights: (B, G, T) per-token weights
        """
        # Scale probabilities by temperature
        scaled_probs = token_probs / self.tau
        
        # Apply sigmoid
        sigmoid_out = torch.sigmoid(scaled_probs)
        
        # Center and scale
        raw_weights = self.alpha * (sigmoid_out - self.mu)
        
        # Clip weights to [L, U]
        weights = torch.clamp(raw_weights, self.weight_lower, self.weight_upper)
        
        return weights
    
    def clip(
        self,
        probs_ratio: torch.Tensor,
        token_probs: torch.Tensor = None,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply weighted trust region clipping.
        
        Args:
            probs_ratio: (B, G, T) token-level probability ratios
            token_probs: (B, G, T) token probabilities (to compute weights)
            weights: (B, G, T) pre-computed weights (optional)
            
        Returns:
            clipped_weighted_ratio: (B, G, T)
        """
        # Compute weights if not provided
        if weights is None:
            if token_probs is None:
                raise ValueError("Must provide either token_probs or weights")
            weights = self.compute_token_weights(token_probs)
        
        # Apply weights to ratios
        weighted_ratio = weights * probs_ratio
        
        # Apply standard clipping to weighted ratios
        # The effective bounds are weight-adjusted
        clipped = torch.clamp(
            weighted_ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        )
        
        return clipped
    
    def clip_with_dynamic_bounds(
        self,
        probs_ratio: torch.Tensor,
        token_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative: Apply dynamic clipping bounds based on weights.
        
        Instead of weighting the ratio, we adjust the clipping bounds
        per token based on confidence.
        
        Args:
            probs_ratio: (B, G, T) token-level probability ratios
            token_probs: (B, G, T) token probabilities
            
        Returns:
            clipped_ratio: (B, G, T)
        """
        weights = self.compute_token_weights(token_probs)
        
        # Dynamic bounds: more confident tokens get tighter bounds
        # Less confident tokens get looser bounds
        epsilon_lower = self.clip_epsilon / weights
        epsilon_upper = self.clip_epsilon * weights
        
        # Per-token clipping
        lower_bound = 1 - epsilon_lower
        upper_bound = 1 + epsilon_upper
        
        clipped = torch.max(torch.min(probs_ratio, upper_bound), lower_bound)
        
        return clipped
