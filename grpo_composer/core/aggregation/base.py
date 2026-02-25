"""
Base Abstract Class for Loss Aggregation in GRPO

Aggregation controls how token-level surrogate losses are reduced to a scalar:
- Token Mean: (1/|o_i|) per-sequence average, then group mean
- Token Sum: sum tokens directly (no length normalization)
- Global Token: 1/Σ|o_i| across all sequences
- Difficulty Weighted: learnable per-difficulty-bin weights
- Group Learnable: power-law length-based weights

The aggregation function receives PRE-COMPUTED per-token losses.
Advantage computation and clipping happen UPSTREAM (not inside aggregation).
"""
from abc import ABC, abstractmethod
import torch


class AggregationFunction(ABC):
    @abstractmethod
    def aggregate(
        self,
        loss_per_token: torch.Tensor,   # (B, T) — pre-computed per-token surrogate losses
        mask: torch.Tensor,             # (B, T) — 1=valid token, 0=padding
        **kwargs,
    ) -> torch.Tensor:
        """
        Reduce per-token losses to a scalar.

        Args:
            loss_per_token: (B, T) token-level losses (already clipped, already multiplied by advantage)
            mask: (B, T) valid token mask

        Returns:
            loss: scalar
        """
        pass