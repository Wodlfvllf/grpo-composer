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
        ratio_min = torch.minimum(ratio* advantage.unsqueeze(-1), clipped_ratio* advantage.unsqueeze(-1)) # shape : (B, G, T)


        mean_length = torch.mean(mask.sum(dim=-1), dim=-1) #shape (B,)
        std_length = torch.std(mask.sum(dim=-1), dim=-1) #shape (B,)
        z = (mask.sum(dim=-1) - mean_length.unsqueeze(-1)) / (std_length.unsqueeze(-1) + 1e-8)

        h = 1 + r * z #shape (B, G)
        f_λ = G * torch.softmax(h**lambda_, dim=-1) #shape (B, G) This is called_unified_token_preference

        loss = f_λ.unsqueeze(-1) * ratio_min  # (B, G, 1) * (B, G, T) → (B, G, T)

        loss_per_token = loss * mask

        total_loss = loss_per_token.sum() 
        total_tokens = mask.sum() 
        loss = total_loss / (total_tokens + 1e-8) 

        return loss

