"""
DAPO: Global Token Normalization

Paper: DAPO

Components Changed (from base GRPO):
- Instead of per-sequence 1/|o_i|, uses global token count
- Normalizes by TOTAL tokens across all sequences in group

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

    DAPO:
        L = (1 / Σ_i |o_i|) * Σ_i Σ_t loss_{i,t}

Rationale:
    Each TOKEN contributes equally, regardless of which sequence it's in.
    Avoids bias toward shorter sequences.
"""
import torch
from .base import AggregationFunction


class GlobalTokenAggregation(AggregationFunction):
    def __init__(self):
        super().__init__()

    def aggregate(
        self,
        loss_per_token: torch.Tensor,   # (B, T) — pre-computed per-token surrogate losses
        mask: torch.Tensor,             # (B, T) — 1=valid token, 0=padding
        **kwargs,
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  loss_per_token (B, T), mask (B, T)
            Step 1: total_loss (scalar) — sum ALL tokens across batch and time
            Step 2: total_tokens (scalar) — count ALL valid tokens
            Step 3: loss (scalar) — total_loss / total_tokens

        Key difference: Normalizes by GLOBAL token count across all sequences.
        """
        # Global token normalization (DAPO style)
        # (loss_per_token * mask).sum(): (B, T) → scalar (sum ALL tokens)
        # mask.sum(): (B, T) → scalar (count ALL valid tokens)
        # Division: scalar / scalar → scalar
        total_loss = (loss_per_token * mask).sum()
        total_tokens = mask.sum()
        return total_loss / (total_tokens + 1e-8)
