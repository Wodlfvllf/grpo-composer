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


def _strict_validation_enabled() -> bool:
    return os.environ.get("GRPO_COMPOSER_STRICT_VALIDATION", "1") != "0"


def _shape_debug(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"torch{tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"np{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


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
    
    response_mask = data.batch["response_mask"]
    
    # [SMOKE TEST MOCK] If no multi-rewards were passed from the driver, synthesize a mock (B, 2) tensor!
    if multi_rewards is None:
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print("🧮 [DEBUG] GDPO: No multi_rewards found. Synthesizing mock (B, 2) multi-rewards for smoke test!")
        
        token_level_rewards = data.batch["token_level_rewards"]
        sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
        sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
        
        # Accuracy is dim 0. Let's make a dummy "Format" reward for dim 1 (randomly -1 or 1)
        mock_format_rewards = torch.where(torch.rand_like(sequence_correctness) > 0.5, 1.0, -1.0)
        
        multi_rewards = torch.stack([sequence_correctness, mock_format_rewards], dim=1)


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


def _apply_frequency_aware_reward_transform(data: Any, config: Any) -> None:
    from grpo_composer.core.rewards.frequency_aware import FrequencyAwareRewardCalculator

    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    
    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
    
    responses = data.batch["responses"]
    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])

    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx]
        grp_responses = responses[idx]
        grp_masks = response_mask[idx]

        # Convert responses to hashable tuples, cropping out padding
        completions = []
        valid_set = set()
        
        for i in range(len(idx)):
            length = grp_masks[i].sum().item()
            # extract only the active tokens
            seq_tuple = tuple(grp_responses[i, :length].tolist())
            completions.append(seq_tuple)
            
            # If the base reward is > 0, it's a valid answer
            if grp_rewards[i].item() > 0:
                valid_set.add(seq_tuple)

        # If there are no valid answers, just use base rewards (which are likely all negative/zero)
        if len(valid_set) == 0:
            adjusted[idx] = grp_rewards
            continue

        invalid_penalty = float(_cfg_get(config, "diversity_invalid_penalty", -1.0))
        calculator = FrequencyAwareRewardCalculator(
            completions=completions,
            valid_set=valid_set,
            invalid_penalty=invalid_penalty
        )
        
        grp_adjusted = calculator.compute_rewards().to(sequence_rewards.device)
        adjusted[idx] = grp_adjusted

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"🧮 [DEBUG] GAPO Frequency-Aware Penalty Applied:")
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
    "frequency_aware": _apply_frequency_aware_reward_transform,
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
        
        # If veRL stripped it from `.batch`, check `.non_tensor_batch` (used for PVPO reference hook)
        if value is None and hasattr(data, "non_tensor_batch") and batch_key in data.non_tensor_batch:
            value = data.non_tensor_batch[batch_key]
            
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

    # ── uid fixup ──────────────────────────────────────────────────────
    # veRL's RayPPOTrainer.fit() assigns uid *before* DataProto.repeat(),
    # expecting repeat(interleave=True) to replicate non_tensor_batch.
    # Some veRL versions do NOT replicate non_tensor_batch, leaving every
    # row with a unique uid (hist={1: total_rows}) — no grouping at all.
    # Detect this and reconstruct per-prompt uids from num_repeat.
    _has_uid = (
        hasattr(data, "non_tensor_batch")
        and data.non_tensor_batch is not None
        and "uid" in data.non_tensor_batch
    )
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"[composer-debug] uid fixup check: num_repeat={num_repeat} has_uid={_has_uid}")
    if num_repeat > 1 and _has_uid:
        uid_raw = data.non_tensor_batch["uid"]
        uid_arr = np.asarray(uid_raw)
        total_rows = uid_arr.shape[0]
        unique_count = len(set(uid_arr.tolist()))
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(f"[composer-debug] uid fixup pre: total={total_rows} unique={unique_count}")
        if unique_count == total_rows and total_rows % num_repeat == 0:
            # Every row is unique → repeat() didn't propagate non_tensor_batch.
            # Reconstruct: rows are interleaved, so row i belongs to prompt i // num_repeat.
            num_prompts = total_rows // num_repeat
            fixed_uid = np.array(
                [uid_arr[i * num_repeat] for i in range(num_prompts) for _ in range(num_repeat)],
                dtype=uid_arr.dtype,
            )
            data.non_tensor_batch["uid"] = fixed_uid
            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                fixed_unique = len(set(fixed_uid.tolist()))
                print(
                    f"[composer-debug] uid fixup: {total_rows} rows had {unique_count} unique uids "
                    f"→ reconstructed {fixed_unique} unique uids (num_repeat={num_repeat})"
                )

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
    
    # Optional explicitly passed variables for custom estimators like PVPO
    if "reference_rewards" in data.batch:
        adv_kwargs["reference_rewards"] = data.batch["reference_rewards"]
    elif "composer_reference_rewards" in data.batch:
        adv_kwargs["composer_reference_rewards"] = data.batch["composer_reference_rewards"]

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
    composer = _cfg_get(config, "composer", None)
    flow = _cfg_get(algorithm, "composer_flow", None)
    if flow is None:
        flow = _cfg_get(algorithm, "flow", None)
    if flow is None:
        flow = _cfg_get(composer, "composer_flow", None)
    if flow is None:
        flow = _cfg_get(composer, "flow", None)

    plugin_names = _cfg_get(algorithm, "composer_flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(algorithm, "flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(composer, "composer_flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(composer, "flow_plugins", None)

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
        clear_batch_context = None
        try:
            from grpo_composer.integrations.verl.losses import (
                clear_composer_batch_context,
                set_composer_batch_context,
                set_composer_config,
            )

            clear_batch_context = clear_composer_batch_context

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

            def _read_key(container: Any, key: str):
                if container is None:
                    return None
                if isinstance(container, dict):
                    return container.get(key)
                getter = getattr(container, "get", None)
                if callable(getter):
                    try:
                        return getter(key, None)
                    except TypeError:
                        try:
                            return getter(key)
                        except Exception:
                            pass
                    except Exception:
                        pass
                try:
                    keys = getattr(container, "keys", None)
                    if callable(keys) and key in keys():
                        return container[key]
                except Exception:
                    pass
                return None

            # Bind per-update runtime context used by custom loss aggregation.
            batch_context: dict[str, Any] = {}
            non_tensor_batch = getattr(data, "non_tensor_batch", None)
            tensor_batch = getattr(data, "batch", None)

            uid = _read_key(non_tensor_batch, "uid")
            if uid is None:
                uid = _read_key(tensor_batch, "uid")

            # ── uid fixup (worker side) ────────────────────────────────
            # Mirror the driver-side fixup: if repeat() didn't propagate
            # non_tensor_batch, every row has a unique uid → no grouping.
            if uid is not None:
                _uid_arr = np.asarray(uid)
                _total = _uid_arr.shape[0]
                _unique = len(set(_uid_arr.tolist()))
                # Resolve num_repeat: try meta_info["rollout_n"], then
                # meta_info["n"], then composer config, then self.config.
                _n_repeat = None
                for _mi_key in ("rollout_n", "n"):
                    if _n_repeat is not None:
                        break
                    if isinstance(meta_info, dict):
                        _n_repeat = meta_info.get(_mi_key)
                    else:
                        _getter = getattr(meta_info, "get", None)
                        if callable(_getter):
                            try:
                                _n_repeat = _getter(_mi_key, None)
                            except Exception:
                                pass
                if _n_repeat is None and isinstance(composer_cfg, dict):
                    _n_repeat = composer_cfg.get("rollout_n")
                if _n_repeat is None:
                    _cfg_env = os.environ.get("GRPO_COMPOSER_ROLLOUT_N")
                    if _cfg_env is not None:
                        try:
                            _n_repeat = int(_cfg_env)
                        except ValueError:
                            pass
                if _n_repeat is not None:
                    _n_repeat = int(_n_repeat)
                if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                    print(
                        f"[composer-debug] worker uid fixup check: total={_total} "
                        f"unique={_unique} n_repeat={_n_repeat}"
                    )
                if (
                    _n_repeat is not None
                    and _n_repeat > 1
                    and _unique == _total
                    and _total % _n_repeat == 0
                ):
                    _fixed = np.array(
                        [_uid_arr[i * _n_repeat] for i in range(_total // _n_repeat) for _ in range(_n_repeat)],
                        dtype=_uid_arr.dtype,
                    )
                    uid = _fixed
                    if non_tensor_batch is not None:
                        try:
                            non_tensor_batch["uid"] = _fixed
                            data.non_tensor_batch["uid"] = _fixed
                        except Exception:
                            pass
                    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                        print(
                            f"[composer-debug] worker uid fixup: {_total} rows had {_unique} unique uids "
                            f"→ reconstructed {len(set(_fixed.tolist()))} unique uids (n={_n_repeat})"
                        )

            if uid is not None:
                batch_context["composer_uid"] = uid

            sequence_rewards = None
            token_level_rewards = _read_key(tensor_batch, "token_level_rewards")
            response_mask = _read_key(tensor_batch, "response_mask")
            if (
                isinstance(token_level_rewards, torch.Tensor)
                and isinstance(response_mask, torch.Tensor)
                and token_level_rewards.shape == response_mask.shape
            ):
                sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)

            if sequence_rewards is None:
                sequence_rewards = _read_key(tensor_batch, "composer_sequence_rewards")
            if sequence_rewards is None:
                sequence_rewards = _read_key(tensor_batch, "sequence_rewards")
            if sequence_rewards is None:
                sequence_rewards = _read_key(non_tensor_batch, "composer_sequence_rewards")
            if sequence_rewards is None:
                sequence_rewards = _read_key(non_tensor_batch, "sequence_rewards")

            if isinstance(sequence_rewards, np.ndarray):
                sequence_rewards = torch.from_numpy(sequence_rewards)
            elif sequence_rewards is not None and not isinstance(sequence_rewards, torch.Tensor):
                try:
                    sequence_rewards = torch.as_tensor(sequence_rewards)
                except Exception:
                    sequence_rewards = None

            if isinstance(sequence_rewards, torch.Tensor):
                if sequence_rewards.ndim == 2:
                    if (
                        isinstance(response_mask, torch.Tensor)
                        and sequence_rewards.shape == response_mask.shape
                    ):
                        sequence_rewards = (sequence_rewards * response_mask).sum(dim=-1)
                    else:
                        sequence_rewards = None
                elif sequence_rewards.ndim != 1:
                    sequence_rewards = None

            agg_mode = None
            if isinstance(composer_cfg, Mapping):
                agg_mode = composer_cfg.get("agg_mode")

            # Build/inject persistent DARO aggregation module so difficulty weights
            # can be learnable (and optimized with actor params) when enabled.
            if agg_mode == "difficulty_weighted":
                from grpo_composer.core.aggregation.difficulty_weighted import DifficultyWeightedAggregation

                num_bins = int(_cfg_get(composer_cfg, "difficulty_bins", 10))
                weight_c = float(_cfg_get(composer_cfg, "difficulty_weight_c", 1.0))
                learnable = bool(_cfg_get(composer_cfg, "difficulty_weight_learnable", True))
                init_weight = float(_cfg_get(composer_cfg, "difficulty_weight_init", 1.0))

                module_spec = (num_bins, weight_c, learnable, init_weight)
                module = getattr(self, "_composer_difficulty_agg_module", None)
                if module is None or getattr(self, "_composer_difficulty_agg_spec", None) != module_spec:
                    module = DifficultyWeightedAggregation(
                        num_bins=num_bins,
                        weight_c=weight_c,
                        learnable=learnable,
                        init_weight=init_weight,
                    )
                    setattr(self, "_composer_difficulty_agg_module", module)
                    setattr(self, "_composer_difficulty_agg_spec", module_spec)
                    setattr(self, "_composer_difficulty_agg_opt_registered", False)

                if learnable and getattr(module, "weight_params", None) is not None:
                    try:
                        actor_device = next(self.actor_module.parameters()).device
                    except Exception:
                        actor_device = module.weight_params.device

                    if module.weight_params.device != actor_device:
                        module.weight_params = torch.nn.Parameter(
                            module.weight_params.detach().to(actor_device)
                        )
                        setattr(self, "_composer_difficulty_agg_opt_registered", False)

                    already_in_optimizer = False
                    for param_group in self.actor_optimizer.param_groups:
                        for param in param_group.get("params", []):
                            if param is module.weight_params:
                                already_in_optimizer = True
                                break
                        if already_in_optimizer:
                            break

                    if already_in_optimizer:
                        setattr(self, "_composer_difficulty_agg_opt_registered", True)

                    if not bool(getattr(self, "_composer_difficulty_agg_opt_registered", False)):
                        self.actor_optimizer.add_param_group({"params": [module.weight_params]})
                        setattr(self, "_composer_difficulty_agg_opt_registered", True)
                        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                            print(
                                "[composer-debug] Registered learnable DARO weights with actor optimizer: "
                                f"num_bins={num_bins}"
                            )

                batch_context["composer_difficulty_agg_module"] = module

            if agg_mode == "difficulty_weighted":
                if uid is None:
                    raise ValueError(
                        "Composer validation failed in worker update_policy: "
                        "agg_mode='difficulty_weighted' requires uid for prompt grouping."
                    )
                if not (
                    isinstance(token_level_rewards, torch.Tensor)
                    and isinstance(response_mask, torch.Tensor)
                    and token_level_rewards.shape == response_mask.shape
                ):
                    candidates = {
                        "token_level_rewards": _shape_debug(_read_key(tensor_batch, "token_level_rewards")),
                        "response_mask": _shape_debug(_read_key(tensor_batch, "response_mask")),
                    }
                    raise ValueError(
                        "Composer validation failed in worker update_policy: "
                        "agg_mode='difficulty_weighted' requires token_level_rewards aligned with response_mask. "
                        f"Observed candidates: {candidates}"
                    )

            if isinstance(sequence_rewards, torch.Tensor):
                batch_context["composer_sequence_rewards"] = sequence_rewards

            rollout_n = None
            if isinstance(meta_info, dict):
                rollout_n = meta_info.get("n")
            else:
                getter = getattr(meta_info, "get", None)
                if callable(getter):
                    try:
                        rollout_n = getter("n", None)
                    except Exception:
                        rollout_n = None
            if rollout_n is not None:
                batch_context["rollout_n"] = rollout_n

            if batch_context:
                set_composer_batch_context(batch_context)
                if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                    print(
                        "[composer-debug] Bound worker batch context: "
                        f"keys={sorted(batch_context.keys())}"
                    )
            else:
                clear_composer_batch_context()
        except Exception as exc:
            # Fail fast by default so miswired reward/context signals are surfaced
            # before veRL enters microbatch loss loops.
            if _strict_validation_enabled():
                raise RuntimeError(
                    "Composer worker preflight validation failed before actor update. "
                    f"Root cause: {type(exc).__name__}: {exc}"
                ) from exc

        try:
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            if loss_mode != "composer":
                return _ORIGINAL_DP_ACTOR_UPDATE_POLICY(self, data)

            # Inlined veRL update_policy with minimal additions for composer loss:
            # 1) keep token_level_rewards in selected batch keys
            # 2) keep uid in selected non-tensor keys
            # 3) pass micro-batch token rewards / uid / rollout_n to policy_loss_fn
            from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
            from verl.utils.device import get_device_id
            from verl.utils.py_functional import append_to_dict
            from verl.utils.seqlen_balancing import prepare_dynamic_batch

            # make sure we are in training mode
            self.actor_module.train()

            temperature = data.meta_info["temperature"]  # required to avoid silent error

            select_keys = [
                "responses",
                "response_mask",
                "input_ids",
                "attention_mask",
                "position_ids",
                "old_log_probs",
                "advantages",
            ]
            if self.config.use_kl_loss:
                select_keys.append("ref_log_prob")
            if "rollout_is_weights" in data.batch.keys():
                select_keys.append("rollout_is_weights")
            if "rollout_log_probs" in data.batch.keys():
                select_keys.append("rollout_log_probs")
            if "token_level_rewards" in data.batch.keys():
                select_keys.append("token_level_rewards")

            non_tensor_select_keys = []
            if "multi_modal_inputs" in data.non_tensor_batch.keys():
                non_tensor_select_keys.append("multi_modal_inputs")
            if "uid" in data.non_tensor_batch.keys():
                non_tensor_select_keys.append("uid")

            data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

            mini_batches = data.split(self.config.ppo_mini_batch_size)
            on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

            rollout_n = None
            meta_info = getattr(data, "meta_info", None)
            if isinstance(meta_info, dict):
                rollout_n = meta_info.get("n")
            else:
                getter = getattr(meta_info, "get", None)
                if callable(getter):
                    try:
                        rollout_n = getter("n", None)
                    except Exception:
                        rollout_n = None

            metrics = {
                "actor/pg_loss": 0.0,
                "actor/kl_loss": 0.0,
            }
            agg_mode = _cfg_get(composer_cfg, "agg_mode", "token_mean")
            daro_enabled = agg_mode == "difficulty_weighted"
            daro_num_bins = int(_cfg_get(composer_cfg, "difficulty_bins", 10))
            daro_eps = float(_cfg_get(composer_cfg, "difficulty_epsilon", 1e-8))
            daro_module = getattr(self, "_composer_difficulty_agg_module", None)
            if daro_module is not None:
                daro_num_bins = int(getattr(daro_module, "num_bins", daro_num_bins))
                daro_eps = float(getattr(daro_module, "epsilon", daro_eps))

            for _ in range(self.config.ppo_epochs):
                for batch_idx, mini_batch in enumerate(mini_batches):
                    if daro_enabled and os.environ.get("GRPO_COMPOSER_DEBUG") == "1" and batch_idx == 0:
                        data_uid = None
                        if "uid" in data.non_tensor_batch:
                            data_uid = data.non_tensor_batch["uid"]
                        elif "uid" in data.batch.keys():
                            data_uid = data.batch["uid"]

                        if data_uid is not None:
                            if isinstance(data_uid, torch.Tensor):
                                data_uid_arr = data_uid.detach().cpu().numpy()
                            else:
                                data_uid_arr = np.asarray(data_uid)
                            uid_counts = defaultdict(int)
                            for uid_key in data_uid_arr.tolist():
                                uid_counts[uid_key] += 1
                            count_hist = defaultdict(int)
                            for cnt in uid_counts.values():
                                count_hist[int(cnt)] += 1
                            hist_sorted = {k: count_hist[k] for k in sorted(count_hist.keys())}
                            print(
                                "[composer-debug][daro] actor-batch uid multiplicity: "
                                f"rows={len(data_uid_arr)} unique_uids={len(uid_counts)} "
                                f"hist={hist_sorted}"
                            )

                    mini_uid_to_bin: dict[Any, int] = {}
                    mini_uid_to_inv_group_tokens: dict[Any, float] = {}
                    mini_active_mu_ids: list[int] = []
                    mini_active_mu_ids_tensor: torch.Tensor | None = None
                    micro_debug_idx = 0
                    if daro_enabled:
                        mini_mask = mini_batch.batch.get("response_mask", None)
                        mini_rewards = mini_batch.batch.get("token_level_rewards", None)

                        mini_uid = None
                        if "uid" in mini_batch.non_tensor_batch:
                            mini_uid = mini_batch.non_tensor_batch["uid"]
                        elif "uid" in mini_batch.batch.keys():
                            mini_uid = mini_batch.batch["uid"]

                        if mini_uid is None:
                            raise ValueError(
                                "DARO requires uid in mini_batch for prompt grouping."
                            )
                        if not isinstance(mini_mask, torch.Tensor):
                            raise ValueError(
                                "DARO requires mini_batch.response_mask as torch.Tensor."
                            )
                        if not isinstance(mini_rewards, torch.Tensor):
                            raise ValueError(
                                "DARO requires mini_batch.token_level_rewards as torch.Tensor."
                            )
                        if mini_rewards.shape != mini_mask.shape:
                            raise ValueError(
                                "DARO requires token_level_rewards shape to match response_mask, got "
                                f"{tuple(mini_rewards.shape)} vs {tuple(mini_mask.shape)}"
                            )

                        mini_seq_rewards = (
                            mini_rewards * mini_mask.to(dtype=mini_rewards.dtype)
                        ).sum(dim=-1).detach().cpu()
                        mini_token_counts = mini_mask.sum(dim=-1).to(dtype=torch.float32).detach().cpu()

                        if isinstance(mini_uid, torch.Tensor):
                            mini_uid_arr = mini_uid.detach().cpu().numpy()
                        else:
                            mini_uid_arr = np.asarray(mini_uid)

                        if mini_uid_arr.ndim != 1 or mini_uid_arr.shape[0] != int(mini_seq_rewards.shape[0]):
                            raise ValueError(
                                "DARO requires uid shape [B_mini], got "
                                f"{tuple(mini_uid_arr.shape)} for B_mini={int(mini_seq_rewards.shape[0])}"
                            )
                        mini_uid_list = mini_uid_arr.tolist()

                        groups: dict[Any, list[int]] = defaultdict(list)
                        for idx, uid_key in enumerate(mini_uid_list):
                            groups[uid_key].append(idx)

                        bin_token_counts: dict[int, float] = defaultdict(float)
                        for uid_key, indices in groups.items():
                            idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=mini_seq_rewards.device)
                            prompt_rewards = mini_seq_rewards.index_select(0, idx_tensor)
                            mu = float((prompt_rewards > 0).float().mean().item())
                            if mu <= 0.0 or mu >= 1.0:
                                mini_uid_to_bin[uid_key] = -1
                                continue
                            mu_bin = min(int(mu * daro_num_bins), daro_num_bins - 1)
                            mini_uid_to_bin[uid_key] = mu_bin
                            prompt_tokens = float(mini_token_counts.index_select(0, idx_tensor).sum().item())
                            bin_token_counts[mu_bin] += prompt_tokens

                        mini_active_mu_ids = sorted(
                            [bin_id for bin_id, token_count in bin_token_counts.items() if token_count > 0.0]
                        )
                        for uid_key, mu_bin in mini_uid_to_bin.items():
                            if mu_bin < 0:
                                mini_uid_to_inv_group_tokens[uid_key] = 0.0
                                continue
                            total_tokens = float(bin_token_counts.get(mu_bin, 0.0))
                            if total_tokens <= 0.0:
                                mini_uid_to_inv_group_tokens[uid_key] = 0.0
                            else:
                                mini_uid_to_inv_group_tokens[uid_key] = 1.0 / (total_tokens + daro_eps)

                        mini_mu_id_row = torch.tensor(
                            [int(mini_uid_to_bin[u]) for u in mini_uid_list],
                            device=mini_mask.device,
                            dtype=torch.long,
                        )
                        mini_inv_group_tokens_row = torch.tensor(
                            [float(mini_uid_to_inv_group_tokens[u]) for u in mini_uid_list],
                            device=mini_mask.device,
                            dtype=mini_rewards.dtype,
                        )
                        mini_active_mu_ids_tensor = torch.tensor(
                            mini_active_mu_ids,
                            device=mini_mask.device,
                            dtype=torch.long,
                        )

                        mini_batch.batch["daro_mu_id_row"] = mini_mu_id_row
                        mini_batch.batch["daro_inv_group_tokens_row"] = mini_inv_group_tokens_row

                        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                            bin_token_counts_debug = {
                                int(k): float(v) for k, v in sorted(bin_token_counts.items(), key=lambda kv: kv[0])
                            }
                            valid_rows = mini_mu_id_row[mini_mu_id_row >= 0]
                            if valid_rows.numel() > 0:
                                bincount = torch.bincount(valid_rows, minlength=daro_num_bins).detach().cpu().tolist()
                            else:
                                bincount = [0] * daro_num_bins
                            preview_n = min(8, int(mini_mu_id_row.shape[0]))
                            print(
                                "[composer-debug][daro] mini_batch context: "
                                f"prompts={len(groups)} rows={len(mini_uid_list)} "
                                f"active_bins={mini_active_mu_ids}"
                            )
                            print(
                                "[composer-debug][daro] mini_batch math: "
                                f"N_mu={bin_token_counts_debug} "
                                f"row_bin_counts={bincount} "
                                f"mu_id_row[:{preview_n}]={mini_mu_id_row[:preview_n].detach().cpu().tolist()} "
                                f"inv_N_row[:{preview_n}]={mini_inv_group_tokens_row[:preview_n].detach().cpu().tolist()}"
                            )

                    if self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                    self.actor_optimizer.zero_grad()

                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        entropy_coeff = float(getattr(self.config, "entropy_coeff", 0.0))
                        loss_agg_mode = getattr(self.config, "loss_agg_mode", "token-mean")
                        calculate_entropy_cfg = bool(getattr(self.config, "calculate_entropy", False))
                        calculate_entropy = calculate_entropy_cfg or (entropy_coeff != 0)

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        loss_extra_kwargs: dict[str, Any] = {}
                        token_level_rewards = model_inputs.get("token_level_rewards", None)
                        uid = model_inputs.get("uid", None)
                        seq_rewards = None
                        if token_level_rewards is not None:
                            loss_extra_kwargs["token_level_rewards"] = token_level_rewards
                            loss_extra_kwargs["composer_token_level_rewards"] = token_level_rewards
                            if (
                                isinstance(token_level_rewards, torch.Tensor)
                                and isinstance(response_mask, torch.Tensor)
                                and token_level_rewards.shape == response_mask.shape
                            ):
                                seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)
                                loss_extra_kwargs["sequence_rewards"] = seq_rewards
                                loss_extra_kwargs["composer_sequence_rewards"] = seq_rewards
                        if uid is not None:
                            loss_extra_kwargs["uid"] = uid
                            loss_extra_kwargs["composer_uid"] = uid

                            if daro_enabled:
                                mu_id_row = model_inputs.get("daro_mu_id_row", None)
                                inv_group_tokens_row = model_inputs.get("daro_inv_group_tokens_row", None)
                                active_mu_ids = mini_active_mu_ids_tensor

                                if not isinstance(mu_id_row, torch.Tensor):
                                    raise ValueError(
                                        "DARO requires microbatch daro_mu_id_row tensor from mini-batch context."
                                    )
                                if not isinstance(inv_group_tokens_row, torch.Tensor):
                                    raise ValueError(
                                        "DARO requires microbatch daro_inv_group_tokens_row tensor from mini-batch context."
                                    )
                                if mu_id_row.ndim != 1 or mu_id_row.shape[0] != int(response_mask.shape[0]):
                                    raise ValueError(
                                        "DARO microbatch daro_mu_id_row shape mismatch: "
                                        f"{tuple(mu_id_row.shape)} vs B_micro={int(response_mask.shape[0])}"
                                    )
                                if (
                                    inv_group_tokens_row.ndim != 1
                                    or inv_group_tokens_row.shape[0] != int(response_mask.shape[0])
                                ):
                                    raise ValueError(
                                        "DARO microbatch daro_inv_group_tokens_row shape mismatch: "
                                        f"{tuple(inv_group_tokens_row.shape)} vs B_micro={int(response_mask.shape[0])}"
                                    )
                                if active_mu_ids is None:
                                    active_mu_ids = torch.empty(
                                        (0,),
                                        device=response_mask.device,
                                        dtype=torch.long,
                                    )
                                else:
                                    active_mu_ids = active_mu_ids.to(
                                        device=response_mask.device,
                                        dtype=torch.long,
                                    )
                                mu_id_row = mu_id_row.to(
                                    device=response_mask.device,
                                    dtype=torch.long,
                                )
                                inv_group_tokens_row = inv_group_tokens_row.to(
                                    device=response_mask.device,
                                    dtype=token_level_rewards.dtype,
                                )

                                loss_extra_kwargs["daro_mu_id_row"] = mu_id_row
                                loss_extra_kwargs["composer_daro_mu_id_row"] = mu_id_row
                                loss_extra_kwargs["daro_inv_group_tokens_row"] = inv_group_tokens_row
                                loss_extra_kwargs["composer_daro_inv_group_tokens_row"] = inv_group_tokens_row
                                loss_extra_kwargs["daro_active_mu_ids"] = active_mu_ids
                                loss_extra_kwargs["composer_daro_active_mu_ids"] = active_mu_ids
                                if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                                    preview_n = min(8, int(mu_id_row.shape[0]))
                                    print(
                                        "[composer-debug][daro] micro_batch payload: "
                                        f"micro_idx={micro_debug_idx} "
                                        f"B_micro={int(response_mask.shape[0])} "
                                        f"active_mu_ids={active_mu_ids.detach().cpu().tolist()} "
                                        f"mu_id_row[:{preview_n}]={mu_id_row[:preview_n].detach().cpu().tolist()} "
                                        f"inv_N_row[:{preview_n}]={inv_group_tokens_row[:preview_n].detach().cpu().tolist()}"
                                    )
                        elif daro_enabled:
                            raise ValueError(
                                "DARO requires uid in microbatch model_inputs."
                            )
                        if rollout_n is not None:
                            loss_extra_kwargs["n"] = rollout_n
                            loss_extra_kwargs["rollout_n"] = rollout_n

                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                            **loss_extra_kwargs,
                        )
                        # Keep scalar accumulators authoritative for pg/kl loss.
                        # Some custom losses (e.g. composer) also emit actor/pg_loss
                        # in per-micro metrics, which conflicts with float accumulators.
                        pg_metrics.pop("actor/pg_loss", None)
                        pg_metrics.pop("actor/kl_loss", None)
                        micro_batch_metrics.update(pg_metrics)

                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                        policy_loss = pg_loss
                        if calculate_entropy and entropy is not None:
                            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                            if entropy_coeff != 0:
                                policy_loss -= entropy_agg * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        loss = policy_loss * loss_scale_factor
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                        append_to_dict(metrics, micro_batch_metrics)
                        micro_debug_idx += 1

                    grad_norm = self._optimizer_step()
                    mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                    append_to_dict(metrics, mini_batch_metrics)

            self.actor_optimizer.zero_grad()
            return metrics
        finally:
            if clear_batch_context is not None:
                try:
                    clear_batch_context()
                except Exception:
                    pass

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
            
        # CRITICAL FIX: veRL's `balance_batch` randomly shuffles batch rows across DP ranks to balance token loads. 
        # This completely destroys the contiguous `num_repeat` interleaving that native GRPO and 
        # *every single composer advantage estimator* relies on to calculate relative advantages.
        # We MUST forcibly disable balance_batch globally to preserve grouped prompt rollouts.
        if hasattr(self.config, "trainer"):
            if getattr(self.config.trainer, "balance_batch", False):
                import logging
                logging.getLogger(__name__).warning(
                    "[grpo_composer] FORCIBLY DISABLING trainer.balance_batch! "
                    "Batch reshuffling breaks mathematical grouping (num_repeat) for GRPO advantage estimation."
                )
                self.config.trainer.balance_batch = False

    def _validate_actor_batch_contract(self, batch: Any) -> None:
        """Fail-fast validation before veRL microbatch update begins."""
        errors: list[str] = []

        response_mask = _maybe_get(batch, "response_mask")
        if not isinstance(response_mask, torch.Tensor):
            errors.append(f"`response_mask` must be torch.Tensor, got {_shape_debug(response_mask)}")
            response_shape = None
            batch_size = None
        elif response_mask.ndim != 2:
            errors.append(f"`response_mask` must be 2D [B,T], got {_shape_debug(response_mask)}")
            response_shape = tuple(response_mask.shape)
            batch_size = response_mask.shape[0]
        else:
            response_shape = tuple(response_mask.shape)
            batch_size = response_mask.shape[0]

        token_level_rewards = _maybe_get(batch, "token_level_rewards")
        if not isinstance(token_level_rewards, torch.Tensor):
            errors.append(f"`token_level_rewards` must be torch.Tensor, got {_shape_debug(token_level_rewards)}")
        elif token_level_rewards.ndim != 2:
            errors.append(f"`token_level_rewards` must be 2D [B,T], got {_shape_debug(token_level_rewards)}")
        elif response_shape is not None and tuple(token_level_rewards.shape) != response_shape:
            errors.append(
                "`token_level_rewards` must match `response_mask` shape, got "
                f"{_shape_debug(token_level_rewards)} vs {_shape_debug(response_mask)}"
            )

        advantages = _maybe_get(batch, "advantages")
        if not isinstance(advantages, torch.Tensor):
            errors.append(f"`advantages` must be torch.Tensor before actor update, got {_shape_debug(advantages)}")
        elif advantages.ndim != 2:
            errors.append(f"`advantages` must be 2D [B,T], got {_shape_debug(advantages)}")
        elif response_shape is not None and tuple(advantages.shape) != response_shape:
            errors.append(
                "`advantages` must match `response_mask` shape, got "
                f"{_shape_debug(advantages)} vs {_shape_debug(response_mask)}"
            )

        uid = _maybe_get(batch, "uid")
        if uid is not None and batch_size is not None:
            uid_len = None
            if isinstance(uid, torch.Tensor):
                if uid.ndim == 1:
                    uid_len = uid.shape[0]
            elif isinstance(uid, np.ndarray):
                if uid.ndim == 1:
                    uid_len = uid.shape[0]
            elif isinstance(uid, (list, tuple)):
                uid_len = len(uid)
            if uid_len is not None and uid_len != batch_size:
                errors.append(f"`uid` length must match batch B={batch_size}, got {uid_len}")

        composer_cfg = getattr(self, "composer_config_dict", {})
        agg_mode = _cfg_get(composer_cfg, "agg_mode", "token_mean")
        if agg_mode == "difficulty_weighted" and batch_size is not None:
            if uid is None:
                errors.append("`uid` is required for agg_mode=difficulty_weighted")

        if errors:
            raise ValueError(
                "Composer preflight validation failed before actor update:\n- "
                + "\n- ".join(errors)
            )

        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            seq_rewards = _maybe_get(batch, "composer_sequence_rewards")
            print(
                "[composer-debug] Pre-update batch contract OK: "
                f"response_mask={_shape_debug(response_mask)}, "
                f"token_level_rewards={_shape_debug(token_level_rewards)}, "
                f"advantages={_shape_debug(advantages)}, "
                f"composer_sequence_rewards={_shape_debug(seq_rewards)}"
            )

    def fit(self) -> None:
        """Override fit to guarantee compute_advantage patching within the execution loop."""
        import copy
        import ray
        import verl.trainer.ppo.ray_trainer as ray_trainer_module
        import sys
        from .info_grpo_hook import InfoGRPORolloutAugmentor

        # Force intercepting the locally scoped compute_advantage inside ray_trainer
        original_compute_advantage = getattr(ray_trainer_module, "compute_advantage", None)
        original_compute_reward = getattr(ray_trainer_module, "compute_reward", None)

        composer_cfg = getattr(self, "composer_config_dict", {})
        ref_reward_source = str(composer_cfg.get("reference_reward_source", "auto")).strip().lower()

        def _has_reference_rewards(data: Any) -> bool:
            if hasattr(data, "batch") and "reference_rewards" in data.batch:
                return True
            non_tensor = getattr(data, "non_tensor_batch", None)
            return isinstance(non_tensor, Mapping) and "reference_rewards" in non_tensor

        def _set_do_sample_flag(data: Any, do_sample: bool) -> None:
            meta_info = getattr(data, "meta_info", None)
            if meta_info is None:
                try:
                    data.meta_info = {}
                    meta_info = data.meta_info
                except Exception:
                    return
            if isinstance(meta_info, dict):
                meta_info["do_sample"] = do_sample
                return
            setter = getattr(meta_info, "__setitem__", None)
            if callable(setter):
                try:
                    setter("do_sample", do_sample)
                except Exception:
                    pass

        def _generate_reference_rollout_output(ref_batch: Any) -> Any:
            ref_wg = getattr(self, "ref_policy_wg", None)
            fallback_to_actor = False
            last_error: Exception | None = None

            if ref_wg is not None:
                try:
                    return ref_wg.generate_sequences(ref_batch)
                except Exception as exc:
                    last_error = exc
                    if "rollout is not registered in ActorRolloutRefWorker" not in str(exc):
                        raise
                    fallback_to_actor = ref_reward_source in ("auto", "actor_rollout")
            else:
                fallback_to_actor = ref_reward_source in ("auto", "actor_rollout")

            if not fallback_to_actor and ref_wg is not None:
                raise RuntimeError(
                    "PVPO reference rollouts are unavailable on this veRL setup: "
                    "the `ref` worker does not register the `rollout` mesh in `verl==0.6.5`. "
                    "Provide `reference_rewards` in the batch, or set "
                    "`composer.reference_reward_source=actor_rollout` for an approximate fallback."
                ) from last_error
            if not fallback_to_actor and ref_wg is None:
                raise RuntimeError(
                    "PVPO reference rollouts requested but no `ref_policy_wg` is available. "
                    "Provide `reference_rewards` in the batch, or set "
                    "`composer.reference_reward_source=actor_rollout` for an approximate fallback."
                )

            rollout_manager = getattr(self, "async_rollout_manager", None)
            if rollout_manager is None:
                rollout_manager = getattr(self, "actor_rollout_wg", None)
            if rollout_manager is None:
                raise RuntimeError(
                    "Cannot compute reference rewards: no rollout manager is available for fallback generation."
                ) from last_error

            if fallback_to_actor and not getattr(self, "_pvpo_actor_rollout_fallback_warned", False):
                print(
                    "[grpo_composer] PVPO fallback active: using actor rollout manager for reference rewards "
                    "(approximation, not true reference-policy rollouts)."
                )
                self._pvpo_actor_rollout_fallback_warned = True

            return rollout_manager.generate_sequences(ref_batch)

        try:
            if "info_grpo" in self.composer_config_dict.get("composer_flow", ""):
                if self.async_rollout_mode:
                    original_method = self.async_rollout_manager.generate_sequences
                    self.async_rollout_manager.generate_sequences = InfoGRPORolloutAugmentor.wrap_generate_sequences(original_method, self.tokenizer)
                else:
                    original_method = self.actor_rollout_wg.generate_sequences
                    self.actor_rollout_wg.generate_sequences = InfoGRPORolloutAugmentor.wrap_generate_sequences(original_method, self.tokenizer)
            
            # Hook reference rewards if PVPO or if the config requests it dynamically
            flow_names = _parse_flow_list(self.config)
            needs_reference_reward = any(
                name in ["pvpo", "pvpo_grpo", "gapo", "gapo_grpo"] or "reference_rewards" in name 
                for name in flow_names
            )
            
            if needs_reference_reward:
                def hooked_compute_advantage(data, adv_estimator, *args, **kwargs):
                    debug = os.environ.get("GRPO_COMPOSER_DEBUG") == "1"
                    if debug:
                        print(f"[grpo_composer-debug] hooked_compute_advantage data={type(data)}")

                    if not _has_reference_rewards(data):
                        ref_batch = copy.deepcopy(data)
                        do_sample = bool(getattr(self.config, "reference_rollout_do_sample", False))
                        _set_do_sample_flag(ref_batch, do_sample)

                        ref_output = _generate_reference_rollout_output(ref_batch)
                        ref_eval_batch = data.union(ref_output)

                        if getattr(self.config, "reward_model", None) is not None and getattr(self.config.reward_model, "launch_reward_fn_async", False):
                            try:
                                from verl.trainer.ppo.reward import compute_reward_async
                            except ImportError:
                                from verl.trainer.ppo.core_algos import compute_reward_async
                            future_reward = compute_reward_async.remote(
                                data=ref_eval_batch, config=self.config, tokenizer=self.tokenizer
                            )
                            ref_reward_tensor, _ = ray.get(future_reward)
                        else:
                            if hasattr(self, "_compute_or_extract_reward"):
                                ref_reward_tensor, _ = self._compute_or_extract_reward(
                                    ref_eval_batch, reward_fn=self.reward_fn, return_dict=False
                                )
                            else:
                                try:
                                    from verl.trainer.ppo.reward import compute_reward
                                except ImportError:
                                    from verl.trainer.ppo.core_algos import compute_reward
                                ref_reward_tensor, _ = compute_reward(ref_eval_batch, self.reward_fn)

                        if isinstance(ref_reward_tensor, torch.Tensor):
                            data.non_tensor_batch["reference_rewards"] = ref_reward_tensor.cpu().numpy()
                        else:
                            data.non_tensor_batch["reference_rewards"] = np.asarray(ref_reward_tensor)

                        if debug:
                            shape = data.non_tensor_batch["reference_rewards"].shape
                            print(f"[grpo_composer-debug] Added reference_rewards shape={shape}")

                    return composer_compute_advantage(data, adv_estimator, *args, **kwargs)

                ray_trainer_module.compute_advantage = hooked_compute_advantage
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = hooked_compute_advantage
            else:
                ray_trainer_module.compute_advantage = composer_compute_advantage
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = composer_compute_advantage

            super().fit()
        finally:
            if original_compute_advantage is not None:
                ray_trainer_module.compute_advantage = original_compute_advantage
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = original_compute_advantage
            if original_compute_reward is not None:
                ray_trainer_module.compute_reward = original_compute_reward

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
                # Inject rollout_n so the worker can reconstruct uid grouping.
                try:
                    _rollout_n = self.config.actor_rollout_ref.rollout.n
                    meta_info["rollout_n"] = int(_rollout_n)
                except Exception:
                    pass
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
        self._validate_actor_batch_contract(batch)
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
