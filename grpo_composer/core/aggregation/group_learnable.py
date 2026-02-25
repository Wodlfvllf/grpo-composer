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
import torch
import torch.nn as nn
from .base import AggregationFunction


class GroupLearnableAggregation(AggregationFunction):
    def __init__(self, lambda_: float = 0.0, r: float = 0.1111, learnable: bool = True):
        super().__init__()
        if learnable:
            self.lambda_ = nn.Parameter(torch.tensor(lambda_))
            self.r = nn.Parameter(torch.tensor(r))
        else:
            self.lambda_ = torch.tensor(lambda_)
            self.r = torch.tensor(r)

    def aggregate(
        self,
        loss_per_token: torch.Tensor,   # (B, T) — pre-computed per-token surrogate losses
        mask: torch.Tensor,             # (B, T) — 1=valid token, 0=padding
        **kwargs,
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  loss_per_token (B, T), mask (B, T)
            Step 1: seq_lengths (B,) — token counts per sequence
            Step 2: mean_length (scalar), std_length (scalar)
            Step 3: z (B,) — standardized length scores
            Step 4: h (B,) — shifted scores: 1 + r * z
            Step 5: f_λ (B,) — power-law weights: B * softmax(h^λ)
            Step 6: weighted_loss (B, T) — f_λ * loss_per_token
            Step 7: loss (scalar) — global token normalization

        Key difference: Learns per-response weights based on length (λ-GRPO).
        """
        B = loss_per_token.shape[0]

        # Sequence lengths
        # mask.sum(dim=-1): (B, T) → (B,)
        seq_lengths = mask.sum(dim=-1).float()

        # Standardize lengths
        # mean_length: scalar, std_length: scalar
        mean_length = seq_lengths.mean()
        std_length = seq_lengths.std() + 1e-8

        # z: standardized length scores
        # (seq_lengths - mean_length) / std_length: (B,)
        z = (seq_lengths - mean_length) / std_length

        # h: shifted scores
        # 1 + self.r * z: (B,)
        h = 1 + self.r * z

        # f_λ: power-law weights with softmax normalization
        # h ** self.lambda_: (B,)
        # B * softmax(dim=0): (B,) — sums to B
        f_lambda = B * torch.softmax(h ** self.lambda_, dim=0)

        # Apply learnable weights
        # f_lambda.unsqueeze(-1): (B,) → (B, 1)
        # loss_per_token * mask: (B, T)
        # weighted_loss: (B, 1) * (B, T) → (B, T)
        weighted_loss = f_lambda.unsqueeze(-1) * loss_per_token * mask

        # Global token normalization
        # weighted_loss.sum(): (B, T) → scalar
        # mask.sum(): (B, T) → scalar
        total_tokens = mask.sum()
        return weighted_loss.sum() / (total_tokens + 1e-8)
