"""
Multi-Reward Processor (GDPO)

Group reward-Decoupled Normalization Policy Optimization for multi-reward RL.

Key Innovation:
    GRPO:  Normalize(Sum(r₁, r₂, ...))    → Reward collapse
    GDPO:  Sum(Normalize(r₁), Normalize(r₂), ...) → Preserves distinctions

Formula:
    1. Per-reward group normalization:
       A_k^(i,j) = (r_k^(i,j) - μ_k^(i)) / (σ_k^(i) + ε)
    
    2. Weighted aggregation:
       A_sum^(i,j) = Σ_k w_k · A_k^(i,j)
    
    3. Optional batch normalization:
       Â_sum^(i,j) = (A_sum^(i,j) - μ_batch) / (σ_batch + ε)

Note: This class has a different API than RewardCalculator because it
processes multiple reward dimensions, not transforms a single reward.

Input:  torch.Tensor of shape (batch_size, num_completions, num_rewards)
Output: torch.Tensor of shape (batch_size, num_completions)
"""

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for a single reward dimension."""
    name: str
    weight: float = 1.0
    conditioned_on: Optional[str] = None
    condition_threshold: Optional[float] = None


class MultiRewardProcessor:
    """
    GDPO: Decoupled normalization for multi-reward RL.
    
    Normalizes each reward dimension separately within groups, then
    aggregates, preserving per-reward distinctions.
    
    Args:
        reward_configs: List of RewardConfig for each reward dimension
        epsilon: Numerical stability constant (default: 1e-8)
        use_batch_norm: Apply batch-level normalization after aggregation
    
    Example:
        >>> configs = [
        ...     RewardConfig(name="safety", weight=1.0),
        ...     RewardConfig(name="helpfulness", weight=1.0),
        ... ]
        >>> processor = MultiRewardProcessor(configs)
        >>> rewards = torch.rand(4, 8, 2)  # (B=4, G=8, N=2)
        >>> normalized = processor.compute_rewards(rewards)  # (4, 8)
    """
    
    def __init__(
        self,
        reward_configs: List[RewardConfig],
        epsilon: float = 1e-8,
        use_batch_norm: bool = True
    ) -> None:
        self.reward_configs = reward_configs
        self.epsilon = epsilon
        self.use_batch_norm = use_batch_norm
        self.reward_to_idx = {cfg.name: i for i, cfg in enumerate(reward_configs)}
    
    def _apply_conditioning(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Apply reward conditioning: zero out reward if dependency not met.
        
        r_k = 0 if dependency r_l < threshold
        """
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
        return (rewards - group_means) / (group_stds + self.epsilon)
    
    def _aggregate(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Weighted sum across reward dimensions.
        
        r̃_sum^(i,j) = Σ_k w_k · r̃_k^(i,j)
        """
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
        Transform raw multi-dimensional rewards using GDPO normalization.
        
        Args:
            rewards: shape (batch_size, num_completions, num_rewards)
        
        Returns:
            normalized_rewards: shape (batch_size, num_completions)
        """
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        
        # Validate shape
        if rewards.ndim != 3:
            raise ValueError(f"Rewards must be 3D (B, G, N), got shape {rewards.shape}")
        if rewards.shape[2] != len(self.reward_configs):
            raise ValueError(
                f"Expected {len(self.reward_configs)} rewards, got {rewards.shape[2]}"
            )
        
        conditioned = self._apply_conditioning(rewards)
        normalized = self._group_normalize(conditioned)
        aggregated = self._aggregate(normalized)
        
        if self.use_batch_norm:
            return self._batch_normalize(aggregated)
        return aggregated