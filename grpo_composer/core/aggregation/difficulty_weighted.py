"""DARO difficulty-weighted aggregation."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .base import AggregationFunction


class DifficultyWeightedAggregation(AggregationFunction):
    def __init__(
        self,
        num_bins: int = 10,
        weight_c: float = 1.0,
        epsilon: float = 1e-8,
        *,
        learnable: bool = False,
        init_weight: float = 1.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.weight_c = weight_c
        self.epsilon = epsilon
        self.learnable = learnable
        self.init_weight = init_weight
        self.weight_params: nn.Parameter | None = None
        if learnable:
            self.weight_params = nn.Parameter(torch.full((num_bins,), float(init_weight)))

    @staticmethod
    def _debug_enabled() -> bool:
        return os.environ.get("GRPO_COMPOSER_DEBUG") == "1"

    def _resolve_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.learnable and self.weight_params is not None:
            return F.softplus(self.weight_params.to(device=device, dtype=dtype)) + self.epsilon
        return torch.full((self.num_bins,), float(self.weight_c), device=device, dtype=dtype)

    def aggregate(
        self,
        loss_per_token: torch.Tensor,          # (B, T)
        mask: torch.Tensor,                    # (B, T)
        mu_id_row: torch.Tensor | None = None,               # (B,)
        inv_group_tokens_row: torch.Tensor | None = None,    # (B,)
        active_mu_ids: torch.Tensor | None = None,           # (M,)
        **kwargs,
    ) -> torch.Tensor:
        if loss_per_token.shape != mask.shape:
            raise ValueError(f"loss/mask shape mismatch: {loss_per_token.shape} vs {mask.shape}")

        if mu_id_row is None:
            mu_id_row = kwargs.get("daro_mu_id_row")
        if inv_group_tokens_row is None:
            inv_group_tokens_row = kwargs.get("daro_inv_group_tokens_row")
        if active_mu_ids is None:
            active_mu_ids = kwargs.get("daro_active_mu_ids")

        if not isinstance(mu_id_row, torch.Tensor):
            raise ValueError("difficulty_weighted requires daro_mu_id_row (torch.Tensor [B])")
        if not isinstance(inv_group_tokens_row, torch.Tensor):
            raise ValueError("difficulty_weighted requires daro_inv_group_tokens_row (torch.Tensor [B])")
        if mu_id_row.ndim != 1 or mu_id_row.shape[0] != loss_per_token.shape[0]:
            raise ValueError(
                f"daro_mu_id_row must be 1D [B], got {tuple(mu_id_row.shape)} for B={loss_per_token.shape[0]}"
            )
        if inv_group_tokens_row.ndim != 1 or inv_group_tokens_row.shape[0] != loss_per_token.shape[0]:
            raise ValueError(
                "daro_inv_group_tokens_row must be 1D [B], got "
                f"{tuple(inv_group_tokens_row.shape)} for B={loss_per_token.shape[0]}"
            )

        device = loss_per_token.device
        dtype = loss_per_token.dtype
        mu_id_row = mu_id_row.to(device=device, dtype=torch.long)
        inv_group_tokens_row = inv_group_tokens_row.to(device=device, dtype=dtype)

        s_j = (loss_per_token * mask).sum(dim=-1)  # (B,)

        valid_rows = (
            (mu_id_row >= 0)
            & (mu_id_row < self.num_bins)
            & torch.isfinite(inv_group_tokens_row)
            & (inv_group_tokens_row > 0)
        )
        if valid_rows.any():
            mu_valid = mu_id_row[valid_rows]
            inv_valid = inv_group_tokens_row[valid_rows]
            s_valid = s_j[valid_rows]
        else:
            mu_valid = torch.empty((0,), device=device, dtype=torch.long)
            inv_valid = torch.empty((0,), device=device, dtype=dtype)
            s_valid = torch.empty((0,), device=device, dtype=dtype)

        if active_mu_ids is None:
            active_mu_ids = torch.unique(mu_valid, sorted=True)
        elif isinstance(active_mu_ids, torch.Tensor):
            active_mu_ids = active_mu_ids.to(device=device, dtype=torch.long).reshape(-1)
        else:
            active_mu_ids = torch.as_tensor(active_mu_ids, device=device, dtype=torch.long).reshape(-1)

        if active_mu_ids.numel() > 0:
            active_mu_ids = torch.unique(active_mu_ids, sorted=True)
            active_mu_ids = active_mu_ids[(active_mu_ids >= 0) & (active_mu_ids < self.num_bins)]

        all_weights = self._resolve_weights(device=device, dtype=dtype)

        if s_valid.numel() == 0:
            loss_main = s_j.sum() * 0.0
        else:
            row_weights = all_weights.index_select(0, mu_valid)
            loss_main = (row_weights * inv_valid * s_valid).sum()

        if active_mu_ids.numel() == 0:
            loss_reg = s_j.sum() * 0.0
        else:
            active_weights = all_weights.index_select(0, active_mu_ids)
            loss_reg = -torch.log(active_weights).sum()

        total_loss = loss_main + loss_reg

        if self._debug_enabled():
            print(
                "[composer-debug][difficulty_weighted] "
                f"B={loss_per_token.shape[0]} active_rows={int(valid_rows.sum().item())} "
                f"active_bins={active_mu_ids.detach().cpu().tolist()} "
                f"learnable={self.learnable}"
            )
            print(
                "[composer-debug][difficulty_weighted] "
                f"loss_main={float(loss_main.detach().item()):.6f} "
                f"loss_reg={float(loss_reg.detach().item()):.6f} "
                f"total={float(total_loss.detach().item()):.6f}"
            )

        return total_loss
