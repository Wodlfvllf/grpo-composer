"""
λ-GRPO: Learnable Response-Level Weights Based on Length

Paper: λ-GRPO

Components Changed (from base GRPO):
- Replaces uniform 1/G with learnable weights f_λ(o_i)
- Power-law formulation based on normalized length

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i L_i   (uniform weighting)

    λ-GRPO:
        L = (1 / Σ_i |o_i|) * Σ_i f_λ(o_i) * L_i

    Weight computation:
        μ_ℓ = (1/G) * Σ_j |o_j|           # Mean length
        σ_ℓ = std({|o_j|})                 # Length std
        z_i = (|o_i| - μ_ℓ) / (σ_ℓ + ε)   # Standardize
        h_i = 1 + r * z_i                  # Shift
        f_λ(o_i) = G * softmax(h_i^λ)      # Power-law + normalize

    λ values:
        λ = 0  → Uniform (DAPO-like)
        λ < 0  → Penalize length (GRPO-like)
        λ > 0  → Reward length
        λ learnable → Adaptive
"""
from ..advantages import *
from ..clipping import *
import torch 
import torch.nn as nn
from .base import AggregationFunction

class GroupLearnableAggregation(AggregationFunction):
    def __init__(self, lambda_: float = 1.0, r: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.lambda_ = nn.Parameter(torch.tensor(lambda_))
            self.r = nn.Parameter(torch.tensor(r))
        else:
            self.register_buffer('lambda_', torch.tensor(lambda_))
            self.register_buffer('r', torch.tensor(r))
    
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
            Step 5: seq_lengths (B, G), mean_length (B,), std_length (B,), z (B, G), h (B, G), f_λ (B, G)
            Step 6: weighted_loss (B, G, T)
            Step 7: loss_per_token (B, G, T)
            Step 8: total_loss (scalar), total_tokens (scalar), loss (scalar)
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

        # 5. Compute λ-GRPO learnable weights f_λ(o_i)
        # mask.sum(dim=-1): (B, G, T) → (B, G) - token counts per sequence
        seq_lengths = mask.sum(dim=-1)
        
        # mean_length: (B, G) → (B,) - mean length per batch
        # std_length: (B, G) → (B,) - std length per batch
        mean_length = seq_lengths.mean(dim=-1)
        std_length = seq_lengths.std(dim=-1)
        
        # z: standardized length scores
        # seq_lengths: (B, G), mean_length.unsqueeze(-1): (B,) → (B, 1)
        # z: (B, G) - (B, 1) / (B, 1) → (B, G)
        z = (seq_lengths - mean_length.unsqueeze(-1)) / (std_length.unsqueeze(-1) + 1e-8)
        
        # h: shifted scores, h: (B, G)
        h = 1 + self.r * z
        
        # f_λ: power-law weights with softmax normalization
        # h ** self.lambda_: (B, G)
        # softmax(dim=-1): (B, G) → (B, G) (sums to 1 over G)
        # G * softmax: (B, G) - rescaled to sum to G
        f_lambda = G * torch.softmax(h ** self.lambda_, dim=-1)

        # 6. Apply learnable weights
        # f_lambda.unsqueeze(-1): (B, G) → (B, G, 1)
        # ratio_min: (B, G, T)
        # weighted_loss: (B, G, 1) * (B, G, T) → (B, G, T)
        weighted_loss = f_lambda.unsqueeze(-1) * ratio_min

        # 7. Mask padding tokens
        # weighted_loss * mask: (B, G, T) * (B, G, T) → (B, G, T)
        loss_per_token = weighted_loss * mask

        # 8. Global token normalization
        # loss_per_token.sum(): (B, G, T) → scalar
        # mask.sum(): (B, G, T) → scalar
        # Division: scalar / scalar → scalar
        total_loss = loss_per_token.sum()
        total_tokens = mask.sum()
        loss = total_loss / (total_tokens + 1e-8)

        return loss

