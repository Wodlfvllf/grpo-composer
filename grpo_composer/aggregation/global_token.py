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

from ..advantages import *
from ..clipping import *
import torch 
import torch.nn as nn
from .base import AggregationFunction


class GlobalTokenAggregation(AggregationFunction):
    def __init__(self):
        super().__init__()

    def compute_aggregation(
        self,
        rewards: torch.Tensor,          # (B, G)
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        mask: torch.Tensor              # (B, G, T) 1=valid token
    ) -> torch.Tensor:

        B, G, T = log_probs.shape

        # 1. Trajectory-level advantage
        advantage = StandardAdvantageFunction().compute_advantages(rewards)  # (B, G)

        # 2. Token-level ratio
        ratio = torch.exp(log_probs - ref_log_probs)  # (B, G, T)

        # 3. Clip ratio (NOT product)
        clipped_ratio = AsymmetricClippingMechanism().clip(ratio)

        # 4. PPO-style min
        ratio_min = torch.minimum(ratio* advantage.unsqueeze(-1), clipped_ratio* advantage.unsqueeze(-1))

        # 5. Mask padding tokens
        loss_per_token = ratio_min * mask

        # 6. Global token normalization (DAPO style)
        total_loss = loss_per_token.sum()                    # Sum ALL tokens
        total_tokens = mask.sum()                            # Total valid tokens
        loss = total_loss / (total_tokens + 1e-8)            # Normalize by global count

        return loss
