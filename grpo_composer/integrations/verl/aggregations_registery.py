"""Composer aggregation mode registry."""

from __future__ import annotations

import os

import numpy as np
import torch

from grpo_composer.core.aggregation.difficulty_weighted import DifficultyWeightedAggregation
from grpo_composer.core.aggregation.global_token import GlobalTokenAggregation
from grpo_composer.core.aggregation.group_learnable import GroupLearnableAggregation
from grpo_composer.core.aggregation.group_uniform import GroupUniformAggregation
from grpo_composer.core.aggregation.token_mean import TokenMeanAggregation
from grpo_composer.core.aggregation.token_sum import TokenSumAggregation
from grpo_composer.core.aggregation.trajectory_level import TrajectoryLevelAggregation
from grpo_composer.core.aggregation.weighted_token import WeightedTokenAggregation

from .loss_context import config_get as _config_get
from .utils import _as_tensor, _validate_tensor_shape


# ---------------------------------------------------------------------------
# Aggregation registry
# ---------------------------------------------------------------------------


def agg_token_mean(loss_mat, mask, config, **kwargs):
    return TokenMeanAggregation().aggregate(loss_mat, mask)


def agg_token_sum(loss_mat, mask, config, **kwargs):
    return TokenSumAggregation().aggregate(loss_mat, mask)


def agg_global_token(loss_mat, mask, config, **kwargs):
    return GlobalTokenAggregation().aggregate(loss_mat, mask)


def agg_group_uniform(loss_mat, mask, config, **kwargs):
    return GroupUniformAggregation().aggregate(loss_mat, mask)


def agg_trajectory_level(loss_mat, mask, config, **kwargs):
    return TrajectoryLevelAggregation().aggregate(loss_mat, mask)


def agg_weighted_token(loss_mat, mask, config, **kwargs):
    log_probs = kwargs.get("log_prob")
    if log_probs is not None:
        log_probs = _as_tensor(log_probs, name="log_prob", device=mask.device)
        _validate_tensor_shape(log_probs, ndim=(2,), first_dim=mask.shape[0], name="log_prob")
        if log_probs.shape != mask.shape:
            raise ValueError(f"log_prob shape must match mask shape, got {log_probs.shape} vs {mask.shape}")

    agg = WeightedTokenAggregation(
        alpha=_config_get(config, "tr_alpha", 1.0),
        tau=_config_get(config, "tr_tau", 1.0),
        mu=_config_get(config, "tr_mu", 0.5),
    )
    return agg.aggregate(loss_mat, mask, log_probs=log_probs)


def agg_group_learnable(loss_mat, mask, config, **kwargs):
    module = kwargs.get("composer_group_learnable_module")
    if module is None:
        # Safe fallback for formula-only runs. Learnable lambda requires trainer-side optimizer wiring.
        learnable = bool(_config_get(config, "lambda_learnable", False))
        if learnable:
            raise ValueError(
                "agg_mode=group_learnable with lambda_learnable=true requires "
                "trainer-injected 'composer_group_learnable_module'."
            )
        module = GroupLearnableAggregation(
            lambda_=float(_config_get(config, "lambda_init", 0.0)),
            r=float(_config_get(config, "lambda_r", 0.1111)),
            learnable=False,
        )

    return module.aggregate(loss_mat, mask)


def agg_difficulty_weighted(loss_mat, mask, config, **kwargs):
    module = kwargs.get("composer_difficulty_agg_module")
    learnable = bool(_config_get(config, "difficulty_weight_learnable", True))
    init_weight = float(_config_get(config, "difficulty_weight_init", 1.0))
    if module is None:
        if learnable:
            raise ValueError(
                "agg_mode=difficulty_weighted with difficulty_weight_learnable=true requires "
                "trainer-injected 'composer_difficulty_agg_module'."
            )
        module = DifficultyWeightedAggregation(
            num_bins=int(_config_get(config, "difficulty_bins", 10)),
            weight_c=float(_config_get(config, "difficulty_weight_c", 1.0)),
            learnable=False,
            init_weight=init_weight,
        )

    batch_size = mask.shape[0]

    mu_id_row = None
    for key in ("daro_mu_id_row", "composer_daro_mu_id_row", "mu_id_row", "composer_mu_id_row"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1 and candidate.shape[0] == batch_size:
            mu_id_row = candidate.to(dtype=torch.long)
            break

    inv_group_tokens_row = None
    for key in (
        "daro_inv_group_tokens_row",
        "composer_daro_inv_group_tokens_row",
        "inv_group_tokens_row",
        "composer_inv_group_tokens_row",
    ):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1 and candidate.shape[0] == batch_size:
            inv_group_tokens_row = candidate
            break

    active_mu_ids = None
    for key in ("daro_active_mu_ids", "composer_daro_active_mu_ids", "active_mu_ids", "composer_active_mu_ids"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        candidate = _as_tensor(candidate, name=key, device=mask.device)
        if candidate.ndim == 1:
            active_mu_ids = candidate.to(dtype=torch.long)
            break

    if mu_id_row is None or inv_group_tokens_row is None:
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            shape_debug = []
            for key in (
                "daro_mu_id_row",
                "composer_daro_mu_id_row",
                "mu_id_row",
                "composer_mu_id_row",
                "daro_inv_group_tokens_row",
                "composer_daro_inv_group_tokens_row",
                "inv_group_tokens_row",
                "composer_inv_group_tokens_row",
                "daro_active_mu_ids",
                "composer_daro_active_mu_ids",
            ):
                value = kwargs.get(key)
                if value is None:
                    continue
                if isinstance(value, np.ndarray):
                    shape_debug.append(f"{key}=np{tuple(value.shape)}")
                elif isinstance(value, torch.Tensor):
                    shape_debug.append(f"{key}=torch{tuple(value.shape)}")
                else:
                    shape_debug.append(f"{key}={type(value).__name__}")
            print(
                "[composer-debug][difficulty_weighted] "
                f"missing DARO row context for B={batch_size}. candidates={shape_debug}"
            )
        raise ValueError(
            "difficulty_weighted requires DARO row context from trainer. "
            "Expected daro_mu_id_row and daro_inv_group_tokens_row aligned to current microbatch."
        )

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        preview_n = min(8, int(batch_size))
        mu_preview = mu_id_row[:preview_n].detach().cpu().tolist()
        inv_preview = inv_group_tokens_row[:preview_n].detach().cpu().tolist()
        active_preview = (
            active_mu_ids.detach().cpu().tolist() if isinstance(active_mu_ids, torch.Tensor) else []
        )
        print(
            "[composer-debug][difficulty_weighted] adapter inputs: "
            f"loss_shape={tuple(loss_mat.shape)} mask_shape={tuple(mask.shape)} "
            f"mu_shape={tuple(mu_id_row.shape)} inv_shape={tuple(inv_group_tokens_row.shape)} "
            f"active_shape={tuple(active_mu_ids.shape) if isinstance(active_mu_ids, torch.Tensor) else 'None'}"
        )
        print(
            "[composer-debug][difficulty_weighted] adapter preview: "
            f"mu_id_row[:{preview_n}]={mu_preview} "
            f"inv_N_row[:{preview_n}]={inv_preview} "
            f"active_mu_ids={active_preview}"
        )

    return module.aggregate(
        loss_mat,
        mask,
        mu_id_row=mu_id_row,
        inv_group_tokens_row=inv_group_tokens_row,
        active_mu_ids=active_mu_ids,
    )


AGG_REGISTRY = {
    "token_mean": agg_token_mean,
    "token_sum": agg_token_sum,
    "global_token": agg_global_token,
    "weighted_token": agg_weighted_token,
    "group_uniform": agg_group_uniform,
    "trajectory_level": agg_trajectory_level,
    "group_learnable": agg_group_learnable,
    "difficulty_weighted": agg_difficulty_weighted,
}

VERL_AGG_MODES = {
    "verl_token_mean": "token-mean",
    "verl_seq_mean_token_sum": "seq-mean-token-sum",
    "verl_seq_mean_token_mean": "seq-mean-token-mean",
}
