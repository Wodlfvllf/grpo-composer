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
import numpy as np
import torch
import torch.nn as nn
from .base import AggregationFunction


class GroupLearnableAggregation(AggregationFunction):
    def __init__(self, lambda_: float = 0.0, r: float = 0.1111, learnable: bool = True):
        super().__init__()
        r_value = torch.tensor(float(r), dtype=torch.float32)
        if learnable:
            # λ is learnable in λ-GRPO; r is a fixed reducer hyperparameter.
            self.lambda_ = nn.Parameter(torch.tensor(float(lambda_), dtype=torch.float32))
            self.register_buffer("r", r_value)
        else:
            self.lambda_ = torch.tensor(float(lambda_), dtype=torch.float32)
            self.register_buffer("r", r_value)

    @staticmethod
    def _build_uid_groups(uid_like, batch_size: int) -> dict[object, list[int]]:
        if uid_like is None:
            return {0: list(range(batch_size))}

        if isinstance(uid_like, torch.Tensor):
            uid_arr = uid_like.detach().cpu().numpy()
        else:
            uid_arr = np.asarray(uid_like)

        if uid_arr.ndim != 1 or uid_arr.shape[0] != batch_size:
            raise ValueError(f"uid must be 1D shape ({batch_size},), got {uid_arr.shape}")

        groups: dict[object, list[int]] = {}
        for i, key in enumerate(uid_arr.tolist()):
            groups.setdefault(key, []).append(i)
        return groups

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
        batch_size = loss_per_token.shape[0]
        device = loss_per_token.device
        dtype = loss_per_token.dtype

        uid = kwargs.get("composer_uid")
        if uid is None:
            uid = kwargs.get("uid")
        uid_groups = self._build_uid_groups(uid, batch_size)

        seq_lengths = mask.sum(dim=-1).to(dtype=torch.float32)
        lambda_value = self.lambda_
        if isinstance(lambda_value, torch.Tensor):
            lambda_value = lambda_value.to(device=device, dtype=torch.float32)
        r_value = self.r.to(device=device, dtype=torch.float32)

        f_lambda = torch.ones((batch_size,), device=device, dtype=torch.float32)
        for indices in uid_groups.values():
            idx = torch.tensor(indices, device=device, dtype=torch.long)
            grp_lengths = seq_lengths[idx]
            grp_mean = grp_lengths.mean()
            grp_std = grp_lengths.std(unbiased=False).clamp_min(1e-8)
            z = (grp_lengths - grp_mean) / grp_std
            h = (1.0 + r_value * z).clamp_min(1e-6)
            g = torch.pow(h, lambda_value)
            grp_size = float(len(indices))
            f_lambda[idx] = grp_size * torch.softmax(g, dim=0)

        weighted_loss = f_lambda.to(dtype=dtype).unsqueeze(-1) * loss_per_token * mask

        # Global token normalization
        # weighted_loss.sum(): (B, T) → scalar
        # mask.sum(): (B, T) → scalar
        total_tokens = mask.sum()
        return weighted_loss.sum() / (total_tokens + 1e-8)
