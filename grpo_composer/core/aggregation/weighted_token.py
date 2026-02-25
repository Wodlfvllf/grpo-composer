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
from .base import AggregationFunction


class WeightedTokenAggregation(AggregationFunction):
    def __init__(
        self,
        alpha: float = 1.0,
        tau: float = 1.0,
        mu: float = 0.5,
        clip_lower: float = 0.5,
        clip_upper: float = 1.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.mu = mu
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def aggregate(
        self,
        loss_per_token: torch.Tensor,       # (B, T) — pre-computed per-token surrogate losses
        mask: torch.Tensor,                 # (B, T) — 1=valid token, 0=padding
        log_probs: torch.Tensor = None,     # (B, T) — current policy log-probs (for weight computation)
        **kwargs,
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  loss_per_token (B, T), mask (B, T), log_probs (B, T)
            Step 1: token_probs (B, T) — exp(log_probs)
            Step 2: scaled (B, T) — token_probs / τ
            Step 3: weights (B, T) — clip(α * (σ(scaled) - μ), L, U)
            Step 4: weighted_loss (B, T) — loss_per_token * weights * mask
            Step 5: loss (scalar) — global token normalization

        Key difference: Applies probability-based weights to down-weight confident tokens.
        """
        if log_probs is None:
            # Fallback: global token mean without weighting
            total_loss = (loss_per_token * mask).sum()
            return total_loss / (mask.sum() + 1e-8)

        # Token-level confidence weights
        # exp(log_probs): (B, T) → (B, T)
        token_probs = torch.exp(log_probs)

        # Scale by temperature
        # token_probs / τ: (B, T) → (B, T)
        scaled = token_probs / self.tau

        # Compute weights: clip(α * (σ(scaled) - μ), L, U)
        # sigmoid(scaled): (B, T) → (B, T)
        # α * (sigmoid - μ): (B, T) → (B, T)
        # clamp: (B, T) → (B, T)
        weights = torch.clamp(
            self.alpha * (torch.sigmoid(scaled) - self.mu),
            self.clip_lower,
            self.clip_upper,
        )

        # Weighted loss with global token normalization
        # loss_per_token * weights * mask: (B, T) * (B, T) * (B, T) → (B, T)
        # .sum(): (B, T) → scalar
        # / mask.sum(): scalar / scalar → scalar
        weighted_loss = loss_per_token * weights * mask
        return weighted_loss.sum() / (mask.sum() + 1e-8)