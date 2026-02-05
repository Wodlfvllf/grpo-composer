"""
Base GRPO: Uniform Group Weighting

Paper: DeepSeekMath (Base GRPO)

Mathematical Form:
    L = (1/G) * Σ_i L_i

Standard uniform weighting across all G responses in the group.
Each response contributes equally to the loss.
"""
from ..advantages import *
from ..clipping import *
import torch 
import torch.nn as nn
from .base import AggregationFunction

class GroupUniformAggregation(AggregationFunction):
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

        # 6a. Per-sequence mean (token mean)
        token_count = mask.sum(dim=-1)  # (B, G)
        seq_loss = loss_per_token.sum(dim=-1) / (token_count + 1e-8)  # (B, G)

        # 6b. Uniform group mean
        loss = seq_loss.mean()  # scalar

        return loss
