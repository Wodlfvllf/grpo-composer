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
from ..advantages import *
from ..clipping import *
import torch 
import torch.nn as nn
from .base import AggregationFunction

class TokenMeanAggregation(AggregationFunction):
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

        # 6. Token mean per sequence
        token_count = mask.sum(dim=-1)  # (B, G)
        seq_loss = loss_per_token.sum(dim=-1) / (token_count + 1e-8)  # (B, G)

        # 7. Group mean
        loss = seq_loss.mean(dim=-1).mean()

        return loss

            