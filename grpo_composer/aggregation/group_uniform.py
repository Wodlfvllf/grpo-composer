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
        mask: torch.Tensor              # (B, G, T) - 1=valid token, 0=padding
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  rewards (B, G), log_probs (B, G, T), ref_log_probs (B, G, T), mask (B, G, T)
            Step 1: advantage (B, G)
            Step 2: ratio (B, G, T)
            Step 3: clipped_ratio (B, G, T)
            Step 4: ratio_min (B, G, T) = min(ratio * A, clipped * A)
            Step 5: loss_per_token (B, G, T)
            Step 6a: token_count (B, G), seq_loss (B, G)
            Step 6b: loss (scalar) - uniform mean over all sequences
        """
        B, G, T = log_probs.shape

        # 1. Trajectory-level advantage
        # rewards: (B, G) → advantage: (B, G)
        advantage = StandardAdvantageFunction().compute_advantages(rewards)

        # 2. Token-level ratio
        # log_probs - ref_log_probs: (B, G, T) → ratio: (B, G, T)
        ratio = torch.exp(log_probs - ref_log_probs)

        # 3. Clip ratio (NOT product)
        # ratio: (B, G, T) → clipped_ratio: (B, G, T)
        clipped_ratio = AsymmetricClippingMechanism().clip(ratio)

        # 4. PPO-style min
        # advantage.unsqueeze(-1): (B, G) → (B, G, 1) for broadcasting
        # ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # clipped_ratio * advantage: (B, G, T) * (B, G, 1) → (B, G, T)
        # torch.minimum: (B, G, T), (B, G, T) → (B, G, T)
        ratio_min = torch.minimum(
            ratio * advantage.unsqueeze(-1),
            clipped_ratio * advantage.unsqueeze(-1)
        )

        # 5. Mask padding tokens
        # ratio_min * mask: (B, G, T) * (B, G, T) → (B, G, T)
        loss_per_token = ratio_min * mask

        # 6a. Per-sequence mean (token mean within each sequence)
        # mask.sum(dim=-1): (B, G, T) → (B, G)
        # loss_per_token.sum(dim=-1): (B, G, T) → (B, G)
        # seq_loss: (B, G) / (B, G) → (B, G)
        token_count = mask.sum(dim=-1)
        seq_loss = loss_per_token.sum(dim=-1) / (token_count + 1e-8)

        # 6b. Uniform group mean (all B*G sequences weighted equally)
        # seq_loss.mean(): (B, G) → scalar
        loss = seq_loss.mean()

        return loss
