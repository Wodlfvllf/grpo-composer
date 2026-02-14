"""
TR-GRPO: Probability-Weighted Token Aggregation for Sharpness Control

Paper: TR-GRPO

Components Changed (from base GRPO):
- Each token weighted by its probability under current policy
- Down-weights high-probability tokens, reduces gradient norm/sharpness

Mathematical Form:
    Token weight:
        w_{i,t} = clip(α * (σ(π_θ(o_{i,t}) / τ) - μ), L, U)

    Weighted gradient:
        g_{i,t} = γ_{i,t} * w_{i,t} * ∇_θ log π_θ(o_{i,t})

    Where:
        σ = sigmoid function
        τ = temperature
        μ = centering constant
        L, U = clipping bounds

Effect:
    Down-weights already-confident tokens.
    Reduces sharpness and gradient norm.
    More stable training.
"""
import torch
import torch.nn as nn
from .base import AggregationFunction
from ..advantages import StandardAdvantageFunction
from ..clipping import AsymmetricClippingMechanism, WeightedTrustRegionClippingMechanism

class WeightedTokenAggregation(AggregationFunction):
    def __init__(self, alpha: float = 1.0, tau: float = 1.0, mu: float = 0.5, clip_lower: float = 0.0, clip_upper: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.mu = mu
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
    
    def compute_aggregation(
        self,   
        rewards: torch.Tensor,          # (B, G)
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        mask: torch.Tensor              # (B, G, T) - 1=valid token, 0=padding
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  rewards (B, G), log_probs (B, G, T), ref_log_probs (B, G, T), mask (B, G, T)
            Step 1: advantage (B, G)
            Step 2: ratio (B, G, T)
            Step 3: clipped_weighted_ratio (B, G, T) - computed by clipping mechanism
            Step 4: weighted_ratio (B, G, T) - we need this for the unclipped term
            Step 5: ratio_min (B, G, T) = min(weighted * A, clipped * A)
            Step 6: loss (scalar) - normalized by valid token count
        
        Key difference: Applies probability-based weights to down-weight confident tokens.
        """
        B, G, T = log_probs.shape

        # 1. Trajectory-level advantage
        # rewards: (B, G) → advantage: (B, G)
        advantage = StandardAdvantageFunction().compute_advantages(rewards)

        # 2. Token-level ratio
        # log_probs - ref_log_probs: (B, G, T) → ratio: (B, G, T)
        ratio = torch.exp(log_probs - ref_log_probs)
        
        # 3. Initialize Clipping Mechanism
        # We pass parameters from the aggregation class
        clipper = WeightedTrustRegionClippingMechanism(
            alpha=self.alpha,
            tau=self.tau,
            mu=self.mu,
            weight_lower=self.clip_lower,
            weight_upper=self.clip_upper
        )
        
        # 4. Compute Weights & Clipped Ratio via Mechanism
        # Need token probabilities for weight computation
        token_probs = torch.exp(log_probs)
        
        # Get weights explicitly to apply to the unclipped term (standard PPO behavior)
        # weights: (B, G, T)
        weights = clipper.compute_token_weights(token_probs) * mask
        
        # Apply weights to ratio (Unclipped term)
        # weighted_ratio: (B, G, T)
        weighted_ratio = ratio * weights
        
        # Get clipped term
        # pass weights to avoid re-computation
        # clipped_ratio: (B, G, T)
        clipped_ratio = clipper.clip(
            probs_ratio=ratio, 
            weights=weights
        )

        # 5. PPO-style min
        # advantage.unsqueeze(-1): (B, G) → (B, G, 1) for broadcasting
        # weighted_ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # clipped_ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # torch.minimum: (B, G, T), (B, G, T) → (B, G, T)
        ratio_min = torch.minimum(
            weighted_ratio * advantage.unsqueeze(-1),
            clipped_ratio * advantage.unsqueeze(-1)
        )

        # 6. Global token normalization
        # (ratio_min * mask).sum(): (B, G, T) → scalar
        # mask.sum(): (B, G, T) → scalar
        # Division: scalar / scalar → scalar
        loss = (ratio_min * mask).sum() / (mask.sum() + 1e-8)

        return loss
        
        