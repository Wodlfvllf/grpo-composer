"""
Base GRPO: Uniform Group Weighting

Paper: DeepSeekMath (Base GRPO)

Mathematical Form:
    L = (1/G) * Σ_i L_i

Standard uniform weighting across all G responses in the group.
Each response contributes equally to the loss.
"""
import torch
from .base import AggregationFunction


class GroupUniformAggregation(AggregationFunction):
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
            Step 1: token_count (B,) — count valid tokens per sequence
            Step 2: seq_loss (B,) — mean over tokens per sequence
            Step 3: loss (scalar) — flat mean over all B sequences (uniform group weight)
        """
        # Per-sequence mean (token mean within each sequence)
        # mask.sum(dim=-1): (B, T) → (B,)
        # (loss_per_token * mask).sum(dim=-1): (B, T) → (B,)
        # seq_loss: (B,) / (B,) → (B,)
        token_count = mask.sum(dim=-1)
        seq_loss = (loss_per_token * mask).sum(dim=-1) / (token_count + 1e-8)

        # Uniform group mean (all B sequences weighted equally)
        # seq_loss.mean(): (B,) → scalar
        return seq_loss.mean()
