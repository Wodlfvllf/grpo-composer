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
from ..clipping import AsymmetricClippingMechanism

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
            Step 3: prob (B, G, T), weight (B, G, T) - computed with no_grad
            Step 4: weighted_ratio (B, G, T)
            Step 5: clipped_ratio (B, G, T)
            Step 6: ratio_min (B, G, T) = min(weighted * A, clipped * A)
            Step 7: loss (scalar) - normalized by valid token count
        
        Key difference: Applies probability-based weights to down-weight confident tokens.
        """
        B, G, T = log_probs.shape

        # 1. Trajectory-level advantage
        # rewards: (B, G) → advantage: (B, G)
        advantage = StandardAdvantageFunction().compute_advantages(rewards)

        # 2. Token-level ratio
        # log_probs - ref_log_probs: (B, G, T) → ratio: (B, G, T)
        ratio = torch.exp(log_probs - ref_log_probs)
        
        # 3. Compute probability-based token weights (no gradient!)
        # This implements: w_{i,t} = clip(α * (σ(π_θ / τ) - μ), L, U)
        with torch.no_grad():
            # prob = π_θ(o_{i,t}): probability under current policy
            # exp(log_probs): (B, G, T) → (B, G, T)
            prob = torch.exp(log_probs)
            
            # Sigmoid scaling with temperature: σ(prob / τ)
            # prob / tau: (B, G, T) → (B, G, T)
            # sigmoid: (B, G, T) → (B, G, T)
            # α * (sigmoid - μ): (B, G, T)
            weight = self.alpha * (torch.sigmoid(prob / self.tau) - self.mu)
            
            # Clip weights to [L, U] bounds
            # weight: (B, G, T) → (B, G, T)
            weight = torch.clamp(weight, self.clip_lower, self.clip_upper)
            
            # Mask padding tokens
            # weight * mask: (B, G, T) * (B, G, T) → (B, G, T)
            weight = weight * mask
        
        # 4. Apply weights to ratio
        # ratio * weight: (B, G, T) * (B, G, T) → (B, G, T)
        weighted_ratio = ratio * weight
        
        # 5. Clip the weighted ratio
        # weighted_ratio: (B, G, T) → clipped_ratio: (B, G, T)
        clipped_ratio = AsymmetricClippingMechanism().clip(weighted_ratio)

        # 6. PPO-style min
        # advantage.unsqueeze(-1): (B, G) → (B, G, 1) for broadcasting
        # weighted_ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # clipped_ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # torch.minimum: (B, G, T), (B, G, T) → (B, G, T)
        ratio_min = torch.minimum(
            weighted_ratio * advantage.unsqueeze(-1),
            clipped_ratio * advantage.unsqueeze(-1)
        )

        # 7. Global token normalization
        # (ratio_min * mask).sum(): (B, G, T) → scalar
        # mask.sum(): (B, G, T) → scalar
        # Division: scalar / scalar → scalar
        loss = (ratio_min * mask).sum() / (mask.sum() + 1e-8)

        return loss
        
        