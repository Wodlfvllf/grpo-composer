"""
Base GRPO: Per-Sequence Token Mean Aggregation

Paper: DeepSeekMath (Base GRPO)

Mathematical Form:
    L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

This is the standard GRPO aggregation:
- First average tokens within each sequence: (1/|o_i|) * Σ_t
- Then average across group: (1/G) * Σ_i

Effect:
    Each sequence contributes equally regardless of length.
    Longer sequences have diluted per-token gradients.
"""
import os
import torch
from .base import AggregationFunction


class TokenMeanAggregation(AggregationFunction):
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
            Step 2: seq_loss (B,) — mean over tokens per sequence: Σ_t loss_t / |o_i|
            Step 3: loss (scalar) — mean over sequences: (1/G) * Σ_i seq_loss_i
        """
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print("🧮 [DEBUG] TokenMeanAggregation: Running base GRPO token_mean math!")

        # Per-sequence token mean: (1/|o_i|) * Σ_t loss_t
        # mask.sum(dim=-1): (B, T) → (B,)
        # (loss_per_token * mask).sum(dim=-1): (B, T) → (B,)
        # seq_loss: (B,) / (B,) → (B,)
        token_count = mask.sum(dim=-1)
        seq_loss = (loss_per_token * mask).sum(dim=-1) / (token_count + 1e-8)

        # Group mean, then batch mean
        # seq_loss.mean(): (B,) → scalar
        total = seq_loss.mean()
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(
                "🧮 [DEBUG] TokenMeanAggregation stats | "
                f"B={loss_per_token.shape[0]} "
                f"token_count(mean/min/max)="
                f"{float(token_count.mean().item()):.2f}/"
                f"{float(token_count.min().item()):.0f}/"
                f"{float(token_count.max().item()):.0f} "
                f"seq_loss(mean/min/max)="
                f"{float(seq_loss.mean().item()):.6f}/"
                f"{float(seq_loss.min().item()):.6f}/"
                f"{float(seq_loss.max().item()):.6f} "
                f"total={float(total.item()):.6f}"
            )
        return total
