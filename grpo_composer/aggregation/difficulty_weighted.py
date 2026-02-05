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
from ..advantages import StandardAdvantageFunction
from ..clipping import AsymmetricClippingMechanism

class DifficultyWeightedAggregation(AggregationFunction):
    def __init__(self, num_bins: int = 10):
        super().__init__()
        self.num_bins = num_bins
        self.weights = nn.Parameter(torch.ones(num_bins))
    
    def compute_aggregation(
        self,   
        rewards: torch.Tensor,          # (B, G)
        log_probs: torch.Tensor,        # (B, G, T)
        ref_log_probs: torch.Tensor,    # (B, G, T)
        mask: torch.Tensor              # (B, G, T) - 1=valid token, 0=padding
    ) -> torch.Tensor:
        """
        Shape Flow:
            Input:  rewards (B, G), log_probs (B, G, T), ...
            Step 1: Group prompts into difficulty bins based on mean reward μ_q
            Step 2: For each bin μ:
                a. Extract subset of prompts (B_μ, G, T)
                b. Compute standard GRPO loss L_μ for this subset
            Step 3: Aggregate: Σ [w_μ * L_μ - ln(w_μ)]
            Step 4: loss (scalar)
        
        Key difference: Learns per-difficulty weights to balance curriculum (DARO).
        """
        B, G, T = log_probs.shape

        groups = [[] for _ in range(self.num_bins)]
        for i in range(B):
            # Assuming rewards are binary (0/1) for correct/incorrect
            difficulty = rewards[i].mean()  # μ_q = proportion of correct responses
            group_idx = min(int(difficulty * self.num_bins), self.num_bins - 1)
            groups[group_idx].append(i)

        total_loss = torch.tensor(0.0, device=log_probs.device, requires_grad=True)
        for group_idx, group in enumerate(groups):
            if len(group) == 0 or group_idx == 0 or group_idx == self.num_bins - 1:
                continue
            group_rewards = rewards[group] #shape : (len(group), G)
            group_log_probs = log_probs[group] #shape : (len(group), G, T)
            group_ref_log_probs = ref_log_probs[group] #shape : (len(group), G, T)
            group_mask = mask[group] #shape : (len(group), G, T)
            group_loss = self._calculate(group_rewards, group_log_probs, group_ref_log_probs, group_mask)
            total_loss += self.weights[group_idx] * group_loss - torch.log(self.weights[group_idx])
        
        return total_loss
        
        
    def _calculate(self, rewards: torch.Tensor, log_probs: torch.Tensor, ref_log_probs: torch.Tensor, mask: torch.Tensor):
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

        # 6. Token mean per sequence
        # mask.sum(dim=-1): (B, G, T) → (B, G)
        # loss_per_token.sum(dim=-1): (B, G, T) → (B, G)
        # seq_loss: (B, G) / (B, G) → (B, G)
        token_count = mask.sum(dim=-1)
        seq_loss = loss_per_token.sum(dim=-1) / (token_count + 1e-8)

        # 7. Group mean, then batch mean
        # seq_loss.mean(dim=-1): (B, G) → (B,)
        # .mean(): (B,) → scalar
        loss = seq_loss.mean(dim=-1).mean()

        return loss
        
