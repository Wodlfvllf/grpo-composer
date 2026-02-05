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
from ..advantages import *
from ..clipping import *
import torch 
import torch.nn as nn
from .base import AggregationFunction

class TokenSumAggregation(AggregationFunction):
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

        # 6. Token sum per sequence
        seq_loss = loss_per_token.sum(dim=-1)

        # 7. Group mean
        loss = seq_loss.mean(dim=-1).mean()

        return loss
