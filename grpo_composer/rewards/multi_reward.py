"""
GDPO: Group reward-Decoupled Normalization Policy Optimization
for Multi-reward RL Optimization
This module implements GDPO's core innovation: decoupled normalization
of multiple rewards to preserve per-reward distinctions and prevent
reward collapse in multi-objective RL training.

Key Innovation:
    GRPO: Normalize(Sum(r₁, r₂, ...))  → Reward collapse
    GDPO: Sum(Normalize(r₁), Normalize(r₂), ...) → Preserves distinctions

Mathematical Formulation:
    For each reward dimension k and rollout j in group i:
    
    A_k^(i,j) = (r_k^(i,j) - μ_k^(i)) / (σ_k^(i) + ε)
    
    A_sum^(i,j) = Σ_k w_k · A_k^(i,j)
    
    Â_sum^(i,j) = (A_sum^(i,j) - μ_batch) / (σ_batch + ε)

Where:
    - i: Prompt/question index (1 to B batch size)
    - j: Rollout index within group (1 to G rollouts per prompt)
    - k: Reward dimension (1 to N rewards)
    - μ_k^(i): Mean of reward k across G rollouts for prompt i
    - σ_k^(i): Std of reward k across G rollouts for prompt i
    - w_k: Weight for reward k (default 1.0)
    - μ_batch, σ_batch: Statistics across entire batch (B×G rollouts)

Benefits over GRPO:
    - Preserves 2^N advantage groups (vs N+1 for GRPO with binary rewards)
    - Faithful multi-reward optimization
    - Better convergence and stability
    - Drop-in replacement for GRPO
"""

import numpy as np
import torch
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    name: str
    weight: float = 1.0
    conditioned_on: Optional[str] = None
    condition_threshold: Optional[float] = None


class GDPORewardProcessor:
    """
    GDPO reward normalization: Normalize each reward separately, then aggregate.
    
    Input:  rewards shape (B, G, N) - B prompts, G rollouts, N rewards
    Output: normalized_rewards shape (B, G) - one value per rollout
    """
    
    def __init__(
        self,
        reward_configs: List[RewardConfig],
        epsilon: float = 1e-8,
        use_batch_norm: bool = True
    ):
        self.reward_configs = reward_configs
        self.epsilon = epsilon
        self.use_batch_norm = use_batch_norm
        self.reward_to_idx = {cfg.name: i for i, cfg in enumerate(reward_configs)}
    
    def _apply_conditioning(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply reward conditioning: r_k = 0 if dependency r_l < threshold"""
        conditioned = rewards.clone()
        
        for i, cfg in enumerate(self.reward_configs):
            if cfg.conditioned_on is not None:
                parent_idx = self.reward_to_idx[cfg.conditioned_on]
                mask = rewards[:, :, parent_idx] >= cfg.condition_threshold
                conditioned[:, :, i] = torch.where(
                    mask,
                    conditioned[:, :, i],
                    torch.zeros_like(conditioned[:, :, i])
                )
        
        return conditioned
    
    def _group_normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize each reward dimension within each prompt's rollout group.
        
        For prompt i, reward k:
            μ_k^(i) = mean(r_k^(i,1), ..., r_k^(i,G))
            σ_k^(i) = std(r_k^(i,1), ..., r_k^(i,G))
            r̃_k^(i,j) = (r_k^(i,j) - μ_k^(i)) / (σ_k^(i) + ε)
        """
        group_means = rewards.mean(dim=1, keepdim=True)  # (B, 1, N)
        group_stds = rewards.std(dim=1, keepdim=True)    # (B, 1, N)
        normalized = (rewards - group_means) / (group_stds + self.epsilon)
        return normalized
    
    def _aggregate(self, normalized: torch.Tensor) -> torch.Tensor:
        """Weighted sum: r̃_sum^(i,j) = Σ_k w_k · r̃_k^(i,j)"""
        B, G, N = normalized.shape
        weights = torch.tensor(
            [cfg.weight for cfg in self.reward_configs],
            dtype=normalized.dtype,
            device=normalized.device
        )
        return (normalized * weights.view(1, 1, N)).sum(dim=2)
    
    def _batch_normalize(self, rewards_sum: torch.Tensor) -> torch.Tensor:
        """
        Batch-level normalization across all rollouts.
        
        r̂_sum^(i,j) = (r̃_sum^(i,j) - μ_batch) / (σ_batch + ε)
        """
        flat = rewards_sum.reshape(-1)
        batch_mean = flat.mean()
        batch_std = flat.std()
        return (rewards_sum - batch_mean) / (batch_std + self.epsilon)
    
    def compute_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Main method: Transform raw rewards using GDPO normalization.
        
        Args:
            rewards: (B, G, N) raw reward values
        
        Returns:
            normalized_rewards: (B, G) normalized reward scalars
        """
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        
        conditioned = self._apply_conditioning(rewards)
        normalized = self._group_normalize(conditioned)
        aggregated = self._aggregate(normalized)
        
        if self.use_batch_norm:
            return self._batch_normalize(aggregated)
        else:
            return aggregated