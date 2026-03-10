"""Custom VERL trainer extensions for grpo_composer.

This module adds a single extensible trainer surface on top of VERL's
RayPPOTrainer and patches module-level compute_advantage to support
composer-specific advantage/reward contexts.
"""

from __future__ import annotations

from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import os
import json
from collections.abc import Mapping
import numpy as np
import torch

from grpo_composer.core.rewards.length_dependent import LengthDependentRewardCalculator
from grpo_composer.core.rewards.multi_reward import MultiRewardProcessor, RewardConfig
from grpo_composer.core.rewards.posterior_composite import PosteriorCompositeRewardCalculator
from grpo_composer.core.rewards.rank_enhanced import RankEnhancedRewardCalculator
from grpo_composer.core.rewards.rts_based import RTSRewardCalculator
from grpo_composer.core.rewards.unlikeliness import UnlikelinessRewardCalculator

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    core_algos = None
    AdvantageEstimator = None
    ray_trainer_module = None
    DataParallelPPOActor = None

    class RayPPOTrainer:  # type: ignore[override]
        """Fallback stub so this module can be imported without verl installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ComposerRayPPOTrainer requires `verl` to be installed. "
                f"Original import error: {_VERL_IMPORT_ERROR!r}"
            )


_ORIGINAL_COMPUTE_ADVANTAGE = None
_ORIGINAL_RAY_TRAINER_CLASS = None
_ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = None
_ORIGINAL_DP_ACTOR_UPDATE_POLICY = None


def _cfg_get(config: Any, key: str, default=None):
    from grpo_composer.integrations.verl.losses import _COMPOSER_CONFIG

    val = None
    if config is not None:
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                val = getter(key, None)
            except TypeError:
                pass
        if val is None:
            val = getattr(config, key, None)

    if val is not None:
        return val

    # Fallback to globally injected composer config
    if _COMPOSER_CONFIG is not None and key in _COMPOSER_CONFIG:
        return _COMPOSER_CONFIG[key]

    return default


def _cfg_get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for part in path:
        if current is None:
            return default
        current = _cfg_get(current, part, None)
    return default if current is None else current


def _maybe_get(data: Any, key: str):
    if hasattr(data, "batch") and key in data.batch.keys():
        return data.batch[key]
    if hasattr(data, "non_tensor_batch") and key in data.non_tensor_batch:
        return data.non_tensor_batch[key]
    return None


def _set_batch_tensor(data: Any, key: str, value: torch.Tensor) -> None:
    if not hasattr(data, "batch"):
        raise ValueError("Data object missing `batch` for tensor assignment")
    data.batch[key] = value


def _set_non_tensor(data: Any, key: str, value: Any) -> None:
    if not hasattr(data, "non_tensor_batch"):
        raise ValueError("Data object missing `non_tensor_batch` for non-tensor assignment")
    data.non_tensor_batch[key] = value


def _get_uid_groups(data: Any, batch_size: int) -> dict[Any, list[int]]:
    uid = None
    if hasattr(data, "non_tensor_batch") and "uid" in data.non_tensor_batch:
        uid = data.non_tensor_batch["uid"]
    elif hasattr(data, "batch") and "uid" in data.batch.keys():
        uid = data.batch["uid"]

    if uid is None:
        return {i: [i] for i in range(batch_size)}

    if isinstance(uid, torch.Tensor):
        uid_arr = uid.detach().cpu().numpy()
    else:
        uid_arr = np.asarray(uid)

    if uid_arr.ndim != 1 or uid_arr.shape[0] != batch_size:
        raise ValueError(f"uid must be shape (bs,), got {uid_arr.shape} for batch size {batch_size}")

    groups: dict[Any, list[int]] = defaultdict(list)
    for i, key in enumerate(uid_arr.tolist()):
        groups[key].append(i)
    return groups


def _sequence_rewards_from_token(token_level_rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    return (token_level_rewards * response_mask).sum(dim=-1)


def _sequence_log_probs_from_token(log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    lengths = response_mask.sum(dim=-1).clamp(min=1)
    return (log_probs * response_mask).sum(dim=-1) / lengths


def _write_sequence_rewards_as_token_rewards(data: Any, sequence_rewards: torch.Tensor) -> None:
    response_mask = data.batch["response_mask"].to(dtype=sequence_rewards.dtype)
    lengths = response_mask.sum(dim=-1).clamp(min=1)
    token_level_rewards = sequence_rewards.unsqueeze(-1) * response_mask / lengths.unsqueeze(-1)
    _set_batch_tensor(data, "token_level_rewards", token_level_rewards)
    _set_batch_tensor(data, "composer_sequence_rewards", sequence_rewards)


def _resolve_sequence_correctness(data: Any, sequence_rewards: torch.Tensor) -> torch.Tensor:
    correctness = _maybe_get(data, "composer_correctness")
    if correctness is None:
        correctness = _maybe_get(data, "correctness")
    if correctness is None:
        correctness = (sequence_rewards > 0).float()

    if isinstance(correctness, np.ndarray):
        correctness = torch.from_numpy(correctness)

    if not isinstance(correctness, torch.Tensor):
        raise ValueError(f"correctness must be torch.Tensor/np.ndarray, got {type(correctness)}")

    if correctness.ndim == 2:
        correctness = correctness.float().mean(dim=-1)
    elif correctness.ndim != 1:
        raise ValueError(f"correctness must be 1D or 2D, got {correctness.shape}")

    if correctness.shape[0] != sequence_rewards.shape[0]:
        raise ValueError(
            f"correctness batch size mismatch: {correctness.shape[0]} vs {sequence_rewards.shape[0]}"
        )

    return (correctness > 0.5).float().to(device=sequence_rewards.device)


def _apply_unlikeliness_reward_transform(data: Any, config: Any) -> None:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    old_log_probs = _maybe_get(data, "old_log_probs")
    if old_log_probs is None:
        return
    if old_log_probs.shape != token_level_rewards.shape:
        raise ValueError(
            f"old_log_probs shape must match token rewards: {old_log_probs.shape} vs {token_level_rewards.shape}"
        )

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
    sequence_log_probs = _sequence_log_probs_from_token(old_log_probs, response_mask)
    beta_rank = float(_cfg_get(config, "unlikeliness_beta", 0.5))

    calculator = UnlikelinessRewardCalculator(beta_rank=beta_rank)
    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])
    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx].unsqueeze(0)
        grp_log_probs = sequence_log_probs[idx].unsqueeze(0)
        grp_adjusted = calculator.compute_rewards(grp_rewards, grp_log_probs).squeeze(0)
        adjusted[idx] = grp_adjusted

    _write_sequence_rewards_as_token_rewards(data, adjusted)


def _apply_rank_enhanced_reward_transform(data: Any, config: Any) -> None:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    old_log_probs = _maybe_get(data, "old_log_probs")
    if old_log_probs is None:
        return

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
    sequence_log_probs = _sequence_log_probs_from_token(old_log_probs, response_mask)

    ranking_method = str(_cfg_get(config, "ranking_method", "weight"))
    ranking_tau = float(_cfg_get(config, "ranking_tau", 0.1))
    calculator = RankEnhancedRewardCalculator(tau=ranking_tau, ranking_method=ranking_method)

    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])
    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx].unsqueeze(0)
        grp_log_probs = sequence_log_probs[idx].unsqueeze(0)
        grp_adjusted = calculator.compute_rewards(grp_rewards, grp_log_probs).squeeze(0)
        adjusted[idx] = grp_adjusted

    _write_sequence_rewards_as_token_rewards(data, adjusted)


def _apply_rts_reward_transform(data: Any, config: Any) -> None:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    rts_scores = _maybe_get(data, "composer_rts_scores")
    if rts_scores is None:
        rts_scores = _maybe_get(data, "rts_scores")
    if rts_scores is None:
        return

    if isinstance(rts_scores, np.ndarray):
        rts_scores = torch.from_numpy(rts_scores)
    if not isinstance(rts_scores, torch.Tensor):
        raise ValueError(f"rts_scores must be tensor/np.ndarray, got {type(rts_scores)}")

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)

    if rts_scores.ndim == 2:
        rts_scores = rts_scores.float().mean(dim=-1)
    elif rts_scores.ndim != 1:
        raise ValueError(f"rts_scores must be 1D or 2D, got {rts_scores.shape}")
    if rts_scores.shape[0] != sequence_rewards.shape[0]:
        raise ValueError(f"rts_scores batch mismatch: {rts_scores.shape[0]} vs {sequence_rewards.shape[0]}")

    beta = float(_cfg_get(config, "rts_beta", 5.0))
    gamma = float(_cfg_get(config, "rts_gamma", 0.5))
    calculator = RTSRewardCalculator(beta=beta, gamma=gamma)

    adjusted = calculator.compute_rewards(
        correctness=sequence_correctness.unsqueeze(0),
        rts_scores=rts_scores.to(device=sequence_rewards.device).unsqueeze(0),
    ).squeeze(0)
    _write_sequence_rewards_as_token_rewards(data, adjusted)


def _apply_posterior_reward_transform(data: Any, config: Any) -> None:
    format_rewards = _maybe_get(data, "composer_format_rewards")
    outcome_rewards = _maybe_get(data, "composer_outcome_rewards")
    thinking_rewards = _maybe_get(data, "composer_thinking_rewards")
    if format_rewards is None or outcome_rewards is None or thinking_rewards is None:
        return

    tensors = []
    for name, value in [
        ("composer_format_rewards", format_rewards),
        ("composer_outcome_rewards", outcome_rewards),
        ("composer_thinking_rewards", thinking_rewards),
    ]:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be tensor/np.ndarray, got {type(value)}")
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1D sequence tensor, got {value.shape}")
        tensors.append(value)

    format_rewards_t, outcome_rewards_t, thinking_rewards_t = [
        t.to(dtype=torch.float32, device=data.batch["response_mask"].device) for t in tensors
    ]

    calculator = PosteriorCompositeRewardCalculator.from_precomputed(
        format_rewards=format_rewards_t,
        outcome_rewards=outcome_rewards_t,
        thinking_rewards=thinking_rewards_t,
    )
    adjusted = calculator.compute_rewards()
    _write_sequence_rewards_as_token_rewards(data, adjusted)


def _apply_multi_reward_transform(data: Any, config: Any) -> None:
    multi_rewards = _maybe_get(data, "composer_multi_rewards")
    if multi_rewards is None:
        multi_rewards = _maybe_get(data, "multi_rewards")
    if multi_rewards is None:
        return

    if isinstance(multi_rewards, np.ndarray):
        multi_rewards = torch.from_numpy(multi_rewards)
    if not isinstance(multi_rewards, torch.Tensor):
        raise ValueError(f"multi_rewards must be tensor/np.ndarray, got {type(multi_rewards)}")

    response_mask = data.batch["response_mask"]
    if multi_rewards.ndim == 3:
        if multi_rewards.shape[:2] != response_mask.shape:
            raise ValueError(
                f"3D multi_rewards must align with response mask shape {response_mask.shape}, got {multi_rewards.shape}"
            )
        multi_seq = (multi_rewards * response_mask.unsqueeze(-1)).sum(dim=1)
    elif multi_rewards.ndim == 2:
        multi_seq = multi_rewards
    else:
        raise ValueError(f"multi_rewards must be 2D or 3D, got {multi_rewards.shape}")

    num_rewards = multi_seq.shape[1]
    weights = _cfg_get(config, "multi_reward_weights", None)
    if weights is None:
        weights = [1.0] * num_rewards
    if len(weights) != num_rewards:
        raise ValueError(f"multi_reward_weights length {len(weights)} must match reward dims {num_rewards}")

    reward_configs = [RewardConfig(name=f"reward_{i}", weight=float(weights[i])) for i in range(num_rewards)]
    processor = MultiRewardProcessor(
        reward_configs=reward_configs,
        use_batch_norm=bool(_cfg_get(config, "multi_reward_use_batch_norm", True)),
    )

    groups = _get_uid_groups(data, multi_seq.shape[0])
    sequence_rewards = torch.zeros(multi_seq.shape[0], device=multi_seq.device, dtype=multi_seq.dtype)
    for indices in groups.values():
        idx = torch.tensor(indices, device=multi_seq.device, dtype=torch.long)
        group_multi = multi_seq[idx].unsqueeze(0)
        processed = processor.compute_rewards(group_multi).squeeze(0)
        sequence_rewards[idx] = processed

    _set_batch_tensor(data, "composer_multi_rewards", multi_seq)
    _write_sequence_rewards_as_token_rewards(data, sequence_rewards)


def _apply_length_dependent_reward_transform(data: Any, config: Any) -> None:
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
    alpha = float(_cfg_get(config, "length_reward_alpha", _cfg_get(config, "length_alpha", 0.05)))
    calculator = LengthDependentRewardCalculator(alpha=alpha)

    lengths = response_mask.sum(dim=-1).long().tolist()
    responses = [list(range(max(1, int(length)))) for length in lengths]
    labels = sequence_correctness.long().cpu()

    adjusted = calculator.compute_rewards(responses=responses, labels=labels).to(sequence_rewards.device)
    _write_sequence_rewards_as_token_rewards(data, adjusted)


def _apply_diversity_adjusted_reward_transform(data: Any, config: Any) -> None:
    from grpo_composer.core.rewards.diversity_adjusted import DiversityAdjustedRewardCalculator

    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)

    # DRA-GRPO uses pairwise similarity of output embeddings.
    # If the generator doesn't emit true hidden state embeddings, we approximate 
    # similarity using a Bag-of-Words feature hash on the generated tokens.
    embeddings = _maybe_get(data, "embeddings")
    if embeddings is None:
        responses = data.batch["responses"]
        hash_bins = 4096
        hashed = responses % hash_bins
        
        # Apply response mask so padding tokens don't contribute to similarity
        pseudo_embeddings = torch.zeros((responses.shape[0], hash_bins), device=responses.device)
        weights = response_mask.float().to(responses.device)
        pseudo_embeddings.scatter_add_(1, hashed, weights)
        embeddings = pseudo_embeddings

    epsilon = float(_cfg_get(config, "diversity_epsilon", 1e-6))
    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])

    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx].unsqueeze(0)
        grp_embeddings = embeddings[idx].unsqueeze(0).float()
        
        calculator = DiversityAdjustedRewardCalculator(
            rewards=grp_rewards, 
            embedding=grp_embeddings, 
            epsilon=epsilon
        )
        grp_adjusted = calculator.compute_rewards().squeeze(0)
        adjusted[idx] = grp_adjusted

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"🧮 [DEBUG] DRA-GRPO Diversity Penalty Applied:")
        print(f"          Original Sequence Rewards (Mean): {sequence_correctness.mean().item():.4f}")
        print(f"          Adjusted Sequence Rewards (Mean): {adjusted.mean().item():.4f}")

    _write_sequence_rewards_as_token_rewards(data, adjusted)


_REWARD_TRANSFORMS = {
    "unlikeliness": _apply_unlikeliness_reward_transform,
    "rank": _apply_rank_enhanced_reward_transform,
    "rts": _apply_rts_reward_transform,
    "posterior": _apply_posterior_reward_transform,
    "multi_reward": _apply_multi_reward_transform,
    "length_dependent": _apply_length_dependent_reward_transform,
    "diversity_adjusted": _apply_diversity_adjusted_reward_transform,
}


def _parse_reward_pipeline(config: Any) -> list[str]:
    pipeline = _cfg_get(config, "composer_reward_pipeline", None)
    if pipeline is None:
        pipeline = _cfg_get(config, "reward_pipeline", None)

    if pipeline is None:
        nested = _cfg_get(config, "composer", None)
        pipeline = _cfg_get(nested, "reward_pipeline", [])

    if isinstance(pipeline, str):
        pipeline = [item.strip() for item in pipeline.split(",") if item.strip()]
    return list(pipeline or [])


def _apply_reward_pipeline(data: Any, config: Any) -> Any:
    pipeline = _parse_reward_pipeline(config)
    
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"🛠️ [DEBUG] Loading Reward Pipeline: {pipeline}")
        
    for transform_name in pipeline:
        transform = _REWARD_TRANSFORMS.get(transform_name)
        if transform is None:
            raise ValueError(
                f"Unknown composer reward transform '{transform_name}'. "
                f"Available: {sorted(_REWARD_TRANSFORMS.keys())}"
            )
            
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(f"🛠️ [DEBUG] Executing Reward Transform: {transform_name}")
            
        transform(data, config)
    return data


def _inject_standard_composer_context(data: Any) -> None:
    token_level_rewards = _maybe_get(data, "token_level_rewards")
    response_mask = _maybe_get(data, "response_mask")
    if isinstance(token_level_rewards, torch.Tensor) and isinstance(response_mask, torch.Tensor):
        if token_level_rewards.shape == response_mask.shape:
            sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
            _set_batch_tensor(data, "composer_sequence_rewards", sequence_rewards)

    # Carry commonly used auxiliary tensors under composer_* aliases.
    for src_key, dst_key in [
        ("old_log_probs", "composer_old_log_probs"),
        ("reward_baselines", "composer_reward_baselines"),
        ("sum_pi_squared", "composer_sum_pi_squared"),
        ("reference_rewards", "composer_reference_rewards"),
        ("multi_rewards", "composer_multi_rewards"),
        ("strata", "composer_strata"),
        ("stratum_ids", "composer_strata"),
        ("log_probs_aug", "composer_log_probs_aug"),
        ("mask_aug", "composer_mask_aug"),
    ]:
        value = _maybe_get(data, src_key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            _set_batch_tensor(data, dst_key, value)
        else:
            _set_non_tensor(data, dst_key, value)


def _collect_adv_optional_kwargs(data: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if hasattr(data, "non_tensor_batch") and "uid" in data.non_tensor_batch:
        kwargs["index"] = data.non_tensor_batch["uid"]

    for batch_key, kwarg_key in [
        ("reward_baselines", "reward_baselines"),
        ("sum_pi_squared", "sum_pi_squared"),
        ("rollout_is_weights", "rollout_is_weights"),
        ("composer_multi_rewards", "multi_rewards"),
        ("multi_rewards", "multi_rewards"),
        ("composer_reference_rewards", "reference_rewards"),
        ("reference_rewards", "reference_rewards"),
        ("composer_old_log_probs", "old_log_probs"),
        ("old_log_probs", "old_log_probs"),
        ("composer_strata", "strata"),
        ("strata", "strata"),
        ("stratum_ids", "strata"),
    ]:
        value = _maybe_get(data, batch_key)
        if value is not None:
            kwargs[kwarg_key] = value

    return kwargs


def composer_compute_advantage(
    data: Any,
    adv_estimator: Any,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[Any] = None,
):
    if core_algos is None or AdvantageEstimator is None or ray_trainer_module is None:
        raise RuntimeError(
            "composer_compute_advantage requires verl to be installed. "
            f"Original import error: {_VERL_IMPORT_ERROR!r}"
        )

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print("🛠️ [DEBUG] composer_compute_advantage IS BEING CALLED SUCCESSFULLY!")

    # Ensure custom advantage estimators are registered in this process
    # (TaskRunner is a separate Ray actor where train_grpo.py's top-level
    # import hasn't run).
    import grpo_composer.integrations.verl.advantages  # noqa: F401

    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = ray_trainer_module.compute_response_mask(data)

    data = _apply_reward_pipeline(data, config)
    _inject_standard_composer_context(data)

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if _cfg_get(config, "use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                _cfg_get(_cfg_get(config, "pf_ppo", None), "reweight_method", None),
                _cfg_get(_cfg_get(config, "pf_ppo", None), "weight_pow", None),
            )
        return data

    if adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs: dict[str, Any] = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index" : data.non_tensor_batch["uid"],
        "config": config,
    }
    adv_kwargs.update(_collect_adv_optional_kwargs(data))

    advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@dataclass
class FlowRuntimeContext:
    """Mutable runtime context shared across flow plugins."""

    metrics: dict[str, Any] = field(default_factory=dict)


class FlowPlugin(ABC):
    """Base class for trainer flow plugins."""

    name: str = "base"

    def configure(self, trainer: "ComposerRayPPOTrainer") -> None:
        return None

    def before_update_actor(self, trainer: "ComposerRayPPOTrainer", batch: Any) -> Any:
        return batch

    def after_update_actor(self, trainer: "ComposerRayPPOTrainer", batch: Any, output: Any) -> Any:
        return output

    def build_loss_context(self, trainer: "ComposerRayPPOTrainer", batch: Any) -> dict[str, Any]:
        return {}


class DefaultFlowPlugin(FlowPlugin):
    name = "default"


class InfoGrpoFlowPlugin(FlowPlugin):
    name = "info_grpo"

    def build_loss_context(self, trainer: "ComposerRayPPOTrainer", batch: Any) -> dict[str, Any]:
        context: dict[str, Any] = {}
        log_probs_aug = _maybe_get(batch, "composer_log_probs_aug")
        mask_aug = _maybe_get(batch, "composer_mask_aug")
        if isinstance(log_probs_aug, torch.Tensor) and isinstance(mask_aug, torch.Tensor):
            if log_probs_aug.shape != mask_aug.shape:
                raise ValueError(
                    f"Info-GRPO aug tensors shape mismatch: {log_probs_aug.shape} vs {mask_aug.shape}"
                )
            context["composer_log_probs_aug"] = log_probs_aug
            context["composer_mask_aug"] = mask_aug
        return context


class PvpoFlowPlugin(FlowPlugin):
    name = "pvpo"

    def before_update_actor(self, trainer: "ComposerRayPPOTrainer", batch: Any) -> Any:
        reference_rewards = _maybe_get(batch, "composer_reference_rewards")
        response_mask = _maybe_get(batch, "response_mask")
        if reference_rewards is None:
            reference_rewards = _maybe_get(batch, "reference_rewards")

        if isinstance(reference_rewards, torch.Tensor):
            if reference_rewards.ndim == 2:
                if response_mask is None or reference_rewards.shape != response_mask.shape:
                    raise ValueError(
                        "PVPO reference_rewards with ndim=2 must match response_mask shape, got "
                        f"{reference_rewards.shape}"
                    )
                reference_rewards = (reference_rewards * response_mask).sum(dim=-1)
            if reference_rewards.ndim != 1:
                raise ValueError(f"PVPO reference_rewards must be 1D/2D tensor, got {reference_rewards.shape}")
            _set_batch_tensor(batch, "composer_reference_rewards", reference_rewards)
        return batch


_FLOW_PLUGIN_REGISTRY: dict[str, type[FlowPlugin]] = {
    DefaultFlowPlugin.name: DefaultFlowPlugin,
    InfoGrpoFlowPlugin.name: InfoGrpoFlowPlugin,
    PvpoFlowPlugin.name: PvpoFlowPlugin,
}


def _parse_flow_list(config: Any) -> list[str]:
    algorithm = _cfg_get(config, "algorithm", None)
    flow = _cfg_get(algorithm, "composer_flow", None)
    if flow is None:
        flow = _cfg_get(algorithm, "flow", None)

    plugin_names = _cfg_get(algorithm, "composer_flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(algorithm, "flow_plugins", None)

    parsed_plugins: list[str] = []
    if isinstance(plugin_names, str):
        parsed_plugins = [name.strip() for name in plugin_names.split(",") if name.strip()]
    elif plugin_names:
        parsed_plugins = [str(name) for name in plugin_names]

    if flow is None:
        flow = "default"
    flow = str(flow)
    if flow and flow not in parsed_plugins:
        parsed_plugins.insert(0, flow)

    if not parsed_plugins:
        parsed_plugins = ["default"]
    return parsed_plugins


def _patch_dp_actor_update_policy() -> None:
    """Patch veRL actor worker update path to bind composer config per batch."""
    if DataParallelPPOActor is None:
        return

    global _ORIGINAL_DP_ACTOR_UPDATE_POLICY
    if _ORIGINAL_DP_ACTOR_UPDATE_POLICY is not None:
        return

    _ORIGINAL_DP_ACTOR_UPDATE_POLICY = DataParallelPPOActor.update_policy

    def _composer_update_policy(self, data):  # type: ignore[override]
        try:
            from grpo_composer.integrations.verl.losses import set_composer_config

            meta_info = getattr(data, "meta_info", None)
            composer_cfg = None
            if isinstance(meta_info, dict):
                composer_cfg = meta_info.get("composer_config")
            else:
                getter = getattr(meta_info, "get", None)
                if callable(getter):
                    composer_cfg = getter("composer_config", None)

            if isinstance(composer_cfg, Mapping):
                composer_cfg = dict(composer_cfg)
            elif composer_cfg is None:
                composer_cfg_json = None
                if isinstance(meta_info, dict):
                    composer_cfg_json = meta_info.get("composer_config_json")
                else:
                    getter = getattr(meta_info, "get", None)
                    if callable(getter):
                        composer_cfg_json = getter("composer_config_json", None)

                if isinstance(composer_cfg_json, str) and composer_cfg_json:
                    try:
                        parsed = json.loads(composer_cfg_json)
                        if isinstance(parsed, dict):
                            composer_cfg = parsed
                    except Exception:
                        composer_cfg = None

            if composer_cfg is None:
                # Reconstruct from primitive meta_info keys if present.
                primitive_cfg = {}
                lookup_keys = ("clip_mode", "agg_mode", "regularizer", "reg_coef")
                if isinstance(meta_info, dict):
                    for k in lookup_keys:
                        meta_key = f"composer_{k}"
                        if meta_key in meta_info and meta_info[meta_key] is not None:
                            primitive_cfg[k] = meta_info[meta_key]
                else:
                    getter = getattr(meta_info, "get", None)
                    if callable(getter):
                        for k in lookup_keys:
                            meta_key = f"composer_{k}"
                            v = getter(meta_key, None)
                            if v is not None:
                                primitive_cfg[k] = v
                if primitive_cfg:
                    composer_cfg = primitive_cfg

            if composer_cfg is None:
                # Last-resort env fallback for worker processes.
                raw = os.environ.get("GRPO_COMPOSER_CONFIG")
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            composer_cfg = parsed
                    except Exception:
                        pass

            if isinstance(composer_cfg, dict) and composer_cfg:
                set_composer_config(composer_cfg)
                if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                    agg = composer_cfg.get("agg_mode", "<missing>")
                    clip = composer_cfg.get("clip_mode", "<missing>")
                    reg = composer_cfg.get("regularizer", "<missing>")
                    print(
                        f"[composer-debug] Bound worker composer config: "
                        f"clip={clip}, agg={agg}, regularizer={reg}"
                    )
            elif os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                keys = []
                if isinstance(meta_info, dict):
                    keys = list(meta_info.keys())
                print(
                    "[composer-debug] Missing composer config in worker update_policy. "
                    f"meta_info_type={type(meta_info)}, meta_info_keys={keys}"
                )
        except Exception:
            # Keep actor update path resilient; if config binding fails, the
            # downstream loss sanity checks will fail fast with a clear error.
            pass

        return _ORIGINAL_DP_ACTOR_UPDATE_POLICY(self, data)

    DataParallelPPOActor.update_policy = _composer_update_policy


# Ensure worker-side config binding when this module is imported via
# actor_rollout_ref.model.external_lib in FSDP worker processes.
_patch_dp_actor_update_policy()


class ComposerRayPPOTrainer(RayPPOTrainer):
    """Single custom trainer that extends VERL's RayPPOTrainer with flow plugins."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inject_composer_config()
        self.composer_context = FlowRuntimeContext()
        self.composer_flow_plugins = self._build_flow_plugins()
        for plugin in self.composer_flow_plugins:
            plugin.configure(self)
        self._validate_supported_modes()

    def _inject_composer_config(self) -> None:
        """Store ``composer:`` YAML keys in a module-level global.

        veRL validates ``actor_rollout_ref.actor`` as ``FSDPActorConfig`` at
        startup, rejecting unknown keys.  We cannot merge composer-specific
        keys (clip_mode, agg_mode, regularizer, …) into the actor OmegaConf
        tree — doing so would break the dataclass conversion in FSDP workers.

        Instead we store them in ``losses._COMPOSER_CONFIG``, which the loss
        function reads via ``_config_get(config, key, default)``'s fallback.
        """
        composer_cfg = _cfg_get(self.config, "composer", None)
        if composer_cfg is None:
            return

        from grpo_composer.integrations.verl.losses import set_composer_config

        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(composer_cfg):
                config_dict = OmegaConf.to_container(composer_cfg, resolve=True)
            else:
                config_dict = dict(composer_cfg) if not isinstance(composer_cfg, dict) else composer_cfg
        except ImportError:
            config_dict = dict(composer_cfg) if not isinstance(composer_cfg, dict) else composer_cfg

        set_composer_config(config_dict)

        # Also store the raw dict so we can explicitly pass it via the DataBatch
        # metadata. Ray workers do not reliably inherit environment variables after
        # initialization, so we push it physically with the data.
        self.composer_config_dict = config_dict
        try:
            os.environ["GRPO_COMPOSER_CONFIG"] = json.dumps(config_dict)
        except Exception:
            pass

    def _build_flow_plugins(self) -> list[FlowPlugin]:
        flow_names = _parse_flow_list(self.config)
        plugins: list[FlowPlugin] = []
        for flow_name in flow_names:
            cls = _FLOW_PLUGIN_REGISTRY.get(flow_name)
            if cls is None:
                raise ValueError(
                    f"Unknown composer flow plugin '{flow_name}'. "
                    f"Available: {sorted(_FLOW_PLUGIN_REGISTRY.keys())}"
                )
            plugins.append(cls())
        return plugins

    def _validate_supported_modes(self) -> None:
        composer_cfg = _cfg_get(self.config, "composer", None)
        agg_mode = _cfg_get(composer_cfg, "agg_mode", "token_mean")
        lambda_learnable = bool(_cfg_get(composer_cfg, "lambda_learnable", False))

        if agg_mode == "group_learnable" and lambda_learnable:
            raise ValueError(
                "group_learnable + lambda_learnable=true needs worker-side optimizer plumbing. "
                "Set lambda_learnable=false for now or extend actor worker update path."
            )

    def fit(self) -> None:
        """Override fit to guarantee compute_advantage patching within the execution loop."""
        import verl.trainer.ppo.ray_trainer as ray_trainer_module
        import sys

        # Force intercepting the locally scoped compute_advantage inside ray_trainer
        original_compute_advantage = getattr(ray_trainer_module, "compute_advantage", None)
        
        try:
            ray_trainer_module.compute_advantage = composer_compute_advantage
            if "verl.trainer.ppo.ray_trainer" in sys.modules:
                sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = composer_compute_advantage
                
            super().fit()
        finally:
            if original_compute_advantage is not None:
                ray_trainer_module.compute_advantage = original_compute_advantage
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = original_compute_advantage

    def _inject_loss_context(self, batch: Any) -> Any:
        _inject_standard_composer_context(batch)
        if hasattr(self, "composer_config_dict"):
            composer_cfg = dict(self.composer_config_dict)
            meta_info = getattr(batch, "meta_info", None)

            if meta_info is None:
                try:
                    batch.meta_info = {}
                    meta_info = batch.meta_info
                except Exception:
                    meta_info = None

            injected = False
            if isinstance(meta_info, dict):
                meta_info["composer_config"] = composer_cfg
                meta_info["composer_config_json"] = json.dumps(composer_cfg)
                # Primitive backup keys for paths that strip nested dict payloads.
                for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                    if k in composer_cfg:
                        meta_info[f"composer_{k}"] = composer_cfg[k]
                injected = True
            else:
                try:
                    meta_info["composer_config"] = composer_cfg
                    meta_info["composer_config_json"] = json.dumps(composer_cfg)
                    for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                        if k in composer_cfg:
                            meta_info[f"composer_{k}"] = composer_cfg[k]
                    injected = True
                except Exception:
                    try:
                        batch.meta_info = dict(meta_info) if meta_info is not None else {}
                        batch.meta_info["composer_config"] = composer_cfg
                        batch.meta_info["composer_config_json"] = json.dumps(composer_cfg)
                        for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                            if k in composer_cfg:
                                batch.meta_info[f"composer_{k}"] = composer_cfg[k]
                        injected = True
                    except Exception:
                        injected = False

            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                keys = []
                current_meta = getattr(batch, "meta_info", None)
                if isinstance(current_meta, dict):
                    keys = list(current_meta.keys())
                print(
                    f"[composer-debug] Inject composer_config into batch.meta_info: "
                    f"success={injected}, meta_info_type={type(current_meta)}, meta_info_keys={keys}"
                )

        for plugin in self.composer_flow_plugins:
            for key, value in plugin.build_loss_context(self, batch).items():
                if isinstance(value, torch.Tensor):
                    _set_batch_tensor(batch, key, value)
                else:
                    _set_non_tensor(batch, key, value)
        return batch

    def _update_actor(self, batch):  # type: ignore[override]
        for plugin in self.composer_flow_plugins:
            batch = plugin.before_update_actor(self, batch)

        batch = self._inject_loss_context(batch)
        output = super()._update_actor(batch)

        for plugin in self.composer_flow_plugins:
            output = plugin.after_update_actor(self, batch, output)
        return output


def patch_verl_main_ppo() -> None:
    """Patch VERL main PPO wiring to use ComposerRayPPOTrainer and compute_advantage."""

    if ray_trainer_module is None:
        raise RuntimeError(
            "patch_verl_main_ppo requires `verl` to be installed. "
            f"Original import error: {_VERL_IMPORT_ERROR!r}"
        )

    global _ORIGINAL_COMPUTE_ADVANTAGE
    global _ORIGINAL_RAY_TRAINER_CLASS
    global _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS

    if _ORIGINAL_COMPUTE_ADVANTAGE is None:
        _ORIGINAL_COMPUTE_ADVANTAGE = ray_trainer_module.compute_advantage
        ray_trainer_module.compute_advantage = composer_compute_advantage

    if _ORIGINAL_RAY_TRAINER_CLASS is None:
        _ORIGINAL_RAY_TRAINER_CLASS = ray_trainer_module.RayPPOTrainer
        ray_trainer_module.RayPPOTrainer = ComposerRayPPOTrainer

    import verl.trainer.main_ppo as main_ppo

    if _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS is None:
        _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = ComposerRayPPOTrainer

    _patch_dp_actor_update_policy()


def unpatch_verl_main_ppo() -> None:
    """Restore VERL's original trainer wiring if it was patched."""

    if ray_trainer_module is None:
        return

    global _ORIGINAL_COMPUTE_ADVANTAGE
    global _ORIGINAL_RAY_TRAINER_CLASS
    global _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS
    global _ORIGINAL_DP_ACTOR_UPDATE_POLICY

    if _ORIGINAL_COMPUTE_ADVANTAGE is not None:
        ray_trainer_module.compute_advantage = _ORIGINAL_COMPUTE_ADVANTAGE
        _ORIGINAL_COMPUTE_ADVANTAGE = None

    if _ORIGINAL_RAY_TRAINER_CLASS is not None:
        ray_trainer_module.RayPPOTrainer = _ORIGINAL_RAY_TRAINER_CLASS
        _ORIGINAL_RAY_TRAINER_CLASS = None

    try:
        import verl.trainer.main_ppo as main_ppo

        if _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS is not None:
            main_ppo.RayPPOTrainer = _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS
            _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = None
    except Exception:
        pass

    if DataParallelPPOActor is not None and _ORIGINAL_DP_ACTOR_UPDATE_POLICY is not None:
        DataParallelPPOActor.update_policy = _ORIGINAL_DP_ACTOR_UPDATE_POLICY
        _ORIGINAL_DP_ACTOR_UPDATE_POLICY = None
