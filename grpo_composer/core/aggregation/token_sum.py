"""
Dr.GRPO: Token Sum Aggregation (No Length Normalization)

Paper: Dr.GRPO

Components Changed (from base GRPO):
- Removes 1/|o_i| per-sequence length normalization
- Sums tokens instead of averaging

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

    Dr.GRPO:
        L = (1/G) * Σ_i Σ_t loss_{i,t}

Rationale:
    Avoids gradient dilution for longer sequences.
    Longer correct responses get proportionally more gradient signal.
"""
import os
import torch
from .base import AggregationFunction


class TokenSumAggregation(AggregationFunction):
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
            Step 1: seq_loss (B,) — SUM over tokens (NO division by token count — key difference)
            Step 2: loss (scalar) — mean over sequences: (1/G) * Σ_i seq_loss_i
        """
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print("🧮 [DEBUG] TokenSumAggregation: Running Dr. GRPO token_sum math!")

        # Per-sequence token SUM (NO length normalization — Dr.GRPO key change)
        # (loss_per_token * mask).sum(dim=-1): (B, T) → (B,)
        seq_loss = (loss_per_token * mask).sum(dim=-1)

        # Group mean, then batch mean
        # seq_loss.mean(): (B,) → scalar
        total = seq_loss.mean()
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            token_count = mask.sum(dim=-1)
            print(
                "🧮 [DEBUG] TokenSumAggregation stats | "
                f"B={loss_per_token.shape[0]} "
                f"token_count(mean/min/max)="
                f"{float(token_count.mean().item()):.2f}/"
                f"{float(token_count.min().item()):.0f}/"
                f"{float(token_count.max().item()):.0f} "
                f"seq_sum(mean/min/max)="
                f"{float(seq_loss.mean().item()):.6f}/"
                f"{float(seq_loss.min().item()):.6f}/"
                f"{float(seq_loss.max().item()):.6f} "
                f"total={float(total.item()):.6f}"
            )
        return total
