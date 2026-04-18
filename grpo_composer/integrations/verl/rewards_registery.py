"""Composer reward transform registry."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import os

import numpy as np
import torch

from grpo_composer.core.rewards.length_dependent import LengthDependentRewardCalculator
from grpo_composer.core.rewards.multi_reward import MultiRewardProcessor, RewardConfig
from grpo_composer.core.rewards.posterior_composite import PosteriorCompositeRewardCalculator
from grpo_composer.core.rewards.rank_enhanced import RankEnhancedRewardCalculator
from grpo_composer.core.rewards.rts_based import RTSRewardCalculator
from grpo_composer.core.rewards.unlikeliness import UnlikelinessRewardCalculator

from .utils import _cfg_get


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

    sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
    sequence_correctness = _resolve_sequence_correctness(data, sequence_rewards)
    ranking_method = str(_cfg_get(config, "ranking_method", "weight"))
    ranking_tau = float(_cfg_get(config, "ranking_tau", 0.1))
    calculator = RankEnhancedRewardCalculator(tau=ranking_tau, ranking_method=ranking_method)

    rrm_ranks = _maybe_get(data, "rrm_ranks")
    if rrm_ranks is None:
        rrm_ranks = _maybe_get(data, "composer_rrm_ranks")

    ranks_tensor: torch.Tensor | None = None
    if rrm_ranks is not None:
        if isinstance(rrm_ranks, np.ndarray):
            rrm_ranks = torch.from_numpy(rrm_ranks)
        if not isinstance(rrm_ranks, torch.Tensor):
            raise ValueError(f"rrm_ranks must be tensor/np.ndarray, got {type(rrm_ranks)}")
        if rrm_ranks.ndim != 1 or rrm_ranks.shape[0] != sequence_rewards.shape[0]:
            raise ValueError(
                f"rrm_ranks must be shape ({sequence_rewards.shape[0]},), got {tuple(rrm_ranks.shape)}"
            )
        ranks_tensor = rrm_ranks.to(device=sequence_rewards.device, dtype=torch.float32)

    sequence_log_probs: torch.Tensor | None = None
    if ranks_tensor is None:
        old_log_probs = _maybe_get(data, "old_log_probs")
        if old_log_probs is None:
            return
        sequence_log_probs = _sequence_log_probs_from_token(old_log_probs, response_mask)

    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])
    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx].unsqueeze(0)
        if ranks_tensor is not None:
            # Map explicit ranks into a monotone proxy so calculator preserves rank order.
            grp_log_probs = -ranks_tensor[idx].unsqueeze(0)
        else:
            grp_log_probs = sequence_log_probs[idx].unsqueeze(0)  # type: ignore[index]
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
    embeddings = _maybe_get(data, "response_hidden_states")
    if embeddings is None:
        # Backward-compatible alias used by earlier trainer/worker patches.
        embeddings = _maybe_get(data, "hidden_states")
    if embeddings is None:
        raise ValueError(
            "No hidden states found in the data. DRA-GRPO requires hidden states to compute pairwise similarity."
        )
    elif embeddings.ndim == 3:
        mask = response_mask.unsqueeze(-1)  # (B, T, 1)
        pooled_embeddings = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif embeddings.ndim == 2:
        pooled_embeddings = embeddings
    else:
        raise ValueError(
            f"Unexpected embedding rank for DRA-GRPO: {tuple(embeddings.shape)}. Expected (B,T,D) or (B,D)."
        )

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(pooled_embeddings.shape)

    epsilon = float(_cfg_get(config, "diversity_epsilon", 1e-6))
    adjusted = torch.zeros_like(sequence_rewards)
    groups = _get_uid_groups(data, sequence_rewards.shape[0])

    for indices in groups.values():
        idx = torch.tensor(indices, device=sequence_rewards.device, dtype=torch.long)
        grp_rewards = sequence_correctness[idx].unsqueeze(0)
        grp_embeddings = pooled_embeddings[idx].unsqueeze(0).float()
        
        calculator = DiversityAdjustedRewardCalculator(
            rewards=grp_rewards, 
            embedding=grp_embeddings, 
            epsilon=epsilon
        )
        grp_adjusted = calculator.compute_rewards().squeeze(0)
        adjusted[idx] = grp_adjusted

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print("🧮 [DEBUG] DRA-GRPO Diversity Penalty Applied:")
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
        print("🧮 [DEBUG] GAPO Frequency-Aware Penalty Applied:")
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
