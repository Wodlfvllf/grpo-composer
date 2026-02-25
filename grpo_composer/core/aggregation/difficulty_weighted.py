"""
DARO: Learnable Per-Difficulty Weights for Curriculum Balancing

Paper: DARO

Components Changed (from base GRPO):
- Groups prompts by difficulty μ_q = (#correct) / G
- Learnable weight w_μ per difficulty bin
- Optimal weights inversely proportional to loss

Mathematical Form:
    Standard GRPO:
        L = uniform average across all prompts

    DARO:
        L = Σ_{μ∈M} [w_μ * L_μ(θ) - ln(w_μ)]

    Difficulty bins:
        M = {1/G, 2/G, ..., (G-1)/G}
        Excludes μ=0 and μ=1 (zero gradient)

    Optimal weights:
        w*_μ ∝ L_μ^{-1}   (inverse loss weighting)

    Per-group normalization:
        Ω_μ = 1/L_μ where L_μ = total tokens in difficulty group

Effect:
    Balances learning across difficulty levels.
    Prevents easy/hard prompts from dominating gradients.
"""
import torch
import torch.nn as nn
from .base import AggregationFunction


class DifficultyWeightedAggregation(AggregationFunction):
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
        self.weights = nn.Parameter(torch.ones(num_bins))

    def aggregate(
        self,
        loss_per_token: torch.Tensor,   # (B, T) — pre-computed per-token surrogate losses
        mask: torch.Tensor,             # (B, T) — 1=valid token, 0=padding
        rewards: torch.Tensor = None,   # (B,) — outcome reward per sequence (for difficulty binning)
        **kwargs,
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  loss_per_token (B, T), mask (B, T), rewards (B,)
            Step 1: Group prompts into difficulty bins based on reward μ_q
            Step 2: For each bin μ:
                a. Extract subset of prompts (B_μ, T)
                b. Compute per-sequence token mean loss L_μ for this subset
            Step 3: Aggregate: Σ [w_μ * L_μ - ln(w_μ)]
            Step 4: loss (scalar)

        Key difference: Learns per-difficulty weights to balance curriculum (DARO).
        """
        if rewards is None:
            # Fallback: token mean if no rewards available for binning
            token_count = mask.sum(dim=-1)
            seq_loss = (loss_per_token * mask).sum(dim=-1) / (token_count + 1e-8)
            return seq_loss.mean()

        B = loss_per_token.shape[0]

        # Per-sequence token mean loss
        # mask.sum(dim=-1): (B, T) → (B,)
        # (loss_per_token * mask).sum(dim=-1): (B, T) → (B,)
        # seq_loss: (B,) / (B,) → (B,)
        token_count = mask.sum(dim=-1)
        seq_loss = (loss_per_token * mask).sum(dim=-1) / (token_count + 1e-8)

        # Bin sequences by difficulty (reward magnitude → difficulty proxy)
        # Higher reward = easier, lower = harder
        # groups[bin_idx] = list of sequence indices in that bin
        groups = [[] for _ in range(self.num_bins)]
        for i in range(B):
            # Assuming rewards are in [0, 1] range for binning
            difficulty = torch.clamp(rewards[i], 0.0, 1.0).item()
            bin_idx = min(int(difficulty * self.num_bins), self.num_bins - 1)
            groups[bin_idx].append(i)

        # Weighted aggregation: Σ [w_μ * L_μ - ln(w_μ)]
        # Excludes μ=0 (all wrong) and μ=num_bins-1 (all correct) — zero gradient
        total_loss = torch.tensor(0.0, device=loss_per_token.device)
        for bin_idx, indices in enumerate(groups):
            if len(indices) == 0 or bin_idx == 0 or bin_idx == self.num_bins - 1:
                continue
            bin_loss = seq_loss[indices].mean()
            total_loss = total_loss + self.weights[bin_idx] * bin_loss - torch.log(self.weights[bin_idx].clamp(min=1e-8))

        return total_loss
