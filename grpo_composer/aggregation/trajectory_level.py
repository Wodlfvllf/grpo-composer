"""
TIC-GRPO: Trajectory-Level Aggregation (Not Token-Level)

Paper: TIC-GRPO

Components Changed (from base GRPO):
- Standard GRPO: Per-token ratios P_θ(s_t|s_{t-1}) / P_θ_old(s_t|s_{t-1})
- TIC-GRPO: Single trajectory ratio P_θ(s_T|s_0) / P_θ_old(s_T|s_0)

Mathematical Form:
    Standard GRPO (token-level):
        ρ_{i,t} = π_θ(o_{i,t}|q, o_{i,<t}) / π_θ_old(o_{i,t}|q, o_{i,<t})

    TIC-GRPO (trajectory-level):
        ρ_i = P_θ(s_T|s_0) / P_θ_old(s_T|s_0)
            = Π_t [π_θ(o_{i,t}|...) / π_θ_old(o_{i,t}|...)]

Benefit:
    Better convergence rate O(log|V| / √N) without MN or σ²_{sT} terms.
    Uses upper-only clipping at trajectory level.
"""
from ..advantages import *
from ..clipping import *
import torch
from .base import AggregationFunction

class TrajectoryLevelAggregation(AggregationFunction):
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
            Step 2: trajectory_ratio (B, G) - product of token ratios = exp(sum of log ratios)
            Step 3: clipped_ratio (B, G) - upper-only clipping
            Step 4: surrogate terms (B, G), ratio_min (B, G)
            Step 5: loss (scalar)
        
        Key difference: Operates at TRAJECTORY level, not token level.
        """
        B, G, T = log_probs.shape

        # 1. Trajectory-level advantage
        # rewards: (B, G) → advantage: (B, G)
        advantage = StandardAdvantageFunction().compute_advantages(rewards)

        # 2. Compute TRAJECTORY-level ratio (not token-level!)
        # log_probs - ref_log_probs: (B, G, T)
        # Mask and sum over T dimension: (B, G, T) → (B, G)
        # exp to get trajectory ratio: (B, G)
        # This computes: Π_t (π_θ / π_old) = exp(Σ_t log(π_θ / π_old))
        log_ratio = log_probs - ref_log_probs
        masked_log_ratio = log_ratio * mask
        trajectory_log_ratio = masked_log_ratio.sum(dim=-1)  # (B, G)
        trajectory_ratio = torch.exp(trajectory_log_ratio)    # (B, G)

        # 3. Clip trajectory ratio (upper-only clipping per TIC-GRPO)
        # trajectory_ratio: (B, G) → clipped_ratio: (B, G)
        clipped_ratio = TrajectoryLevelClippingMechanism().clip(probs_ratio=trajectory_ratio)

        # 4. PPO-style min at trajectory level
        # trajectory_ratio * advantage: (B, G) * (B, G) → (B, G)
        # clipped_ratio * advantage: (B, G) * (B, G) → (B, G)
        # torch.minimum: (B, G), (B, G) → (B, G)
        surrogate1 = trajectory_ratio * advantage
        surrogate2 = clipped_ratio * advantage
        ratio_min = torch.minimum(surrogate1, surrogate2)

        # 5. Mean over all trajectories
        # ratio_min: (B, G) → scalar
        loss = ratio_min.mean()

        return loss