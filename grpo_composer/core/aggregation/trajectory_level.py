"""
TIC-GRPO: Trajectory-Level Aggregation (Not Token-Level)

Paper: TIC-GRPO

Components Changed (from base GRPO):
- Standard GRPO: Per-token ratios P_θ(s_t|s_{t-1}) / P_θ_old(s_t|s_{t-1})
- TIC-GRPO: Single trajectory ratio P_θ(s_T|s_0) / P_θ_old(s_T|s_0)

Mathematical Form:
    Standard GRPO (token-level):
        ρ_{i,t} = π_θ(o_{i,t}|q, o_{i,<t}) / π_θ_old(o_{i,t}|q, o_{i,<t})

    TIC-GRPO (trajectory-level):
        ρ_i = P_θ(s_T|s_0) / P_θ_old(s_T|s_0)
            = Π_t [π_θ(o_{i,t}|...) / π_θ_old(o_{i,t}|...)]

Benefit:
    Better convergence rate O(log|V| / √N) without MN or σ²_{sT} terms.
    Uses upper-only clipping at trajectory level.

Note:
    When used with trajectory-level clipping upstream, the loss_per_token
    may actually be uniform per-sequence (broadcast from trajectory ratio).
    This aggregation simply takes the mean over sequences.
"""
import torch
from .base import AggregationFunction


class TrajectoryLevelAggregation(AggregationFunction):
    def __init__(self):
        super().__init__()

    def aggregate(
        self,
        loss_per_token: torch.Tensor,   # (B, T) — pre-computed surrogate losses
        mask: torch.Tensor,             # (B, T) — 1=valid token, 0=padding
        **kwargs,
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  loss_per_token (B, T), mask (B, T)
            Step 1: token_count (B,) — count valid tokens per sequence
            Step 2: seq_loss (B,) — mean over valid tokens
            Step 3: loss (scalar) — mean over sequences

        Key difference: When paired with trajectory-level clipping,
        the ratio is computed at TRAJECTORY level (product of token ratios),
        so loss_per_token may be uniform per sequence.
        """
        # Token mean per sequence
        # mask.sum(dim=-1): (B, T) → (B,)
        # (loss_per_token * mask).sum(dim=-1): (B, T) → (B,)
        token_count = mask.sum(dim=-1)
        seq_loss = (loss_per_token * mask).sum(dim=-1) / (token_count + 1e-8)

        # Mean over all trajectories
        # seq_loss.mean(): (B,) → scalar
        return seq_loss.mean()