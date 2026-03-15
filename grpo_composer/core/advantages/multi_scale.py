import torch
from typing import Union, List

from .base import AdvantageFunction


class MultiScaleAdvantageFunction(AdvantageFunction):
    """
    MS-GRPO: Multi-Scale Advantage via Dilated Scale Sampling

    Calculates advantages over randomly sampled subgroups of various scales (sizes)
    to reduce variance and prevent combinatorial explosion.
    
    Paper: Multi-Scale Group Relative Policy Optimization (MS-GRPO)
    """

    def __init__(
        self,
        tau_min: int = 2,
        num_scales: int = 4,
        samples_per_scale: int = 4,
        weights: Union[List[float], torch.Tensor] = None,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            tau_min: Minimum scale size (must be >= 2 for std).
            num_scales (M): Number of different subgroup sizes (τ) to sample. 
                            Scales are uniformly distributed between tau_min and G.
            samples_per_scale (K): Number of random subgroups to sample per scale.
            weights: List of weights for each scale. Defaults to uniform.
            epsilon: Small constant to avoid division by zero in variance computation.
        """
        super().__init__()
        self.tau_min = max(tau_min, 2)
        self.num_scales = num_scales
        self.samples_per_scale = samples_per_scale
        self.weights = weights
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-GRPO advantages using Dilated Scale Sampling.
        
        Args:
            rewards: Tensor of shape (1, G) containing sequence scores for a single prompt group.
                     (veRL batches are grouped by prompt in `advantages.py`).
        Returns:
            advantages: Tensor of shape (1, G) containing multi-scale advantages.
        """
        # rewards comes in as (1, G) from advantages.py _compute_groupwise
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(0)
            
        B, G = rewards.shape
        rewards = rewards.unsqueeze(-1)  # (1, G, 1) to match vectorized math

        num_scales = min(self.num_scales, max(1, G - self.tau_min + 1))
        
        # 1. Determine the scales (τ values) to use
        # E.g., if G=16, tau_min=2, num_scales=4, scales might be [2, 6, 11, 16]
        scales = torch.linspace(self.tau_min, G, num_scales, device=rewards.device).long()
        scales = torch.unique(scales) # Handle cases where G is very small
        actual_num_scales = len(scales)

        # 2. Configure weights
        if self.weights is None:
            weights = torch.ones(actual_num_scales, device=rewards.device) / actual_num_scales
        else:
            weights = torch.tensor(self.weights, device=rewards.device, dtype=rewards.dtype)
            if len(weights) > actual_num_scales:
                weights = weights[:actual_num_scales]
            weights = weights / weights.sum()

        device = rewards.device
        dtype = rewards.dtype
        
        all_scale_advantages = []

        # 3. Vectorized Subgroup Sampling (Dilated Scale Sampling)
        for scale_idx, tau in enumerate(scales):
            tau = tau.item()
            K = self.samples_per_scale
            
            # If tau is G, there's only 1 possible subgroup (the full group).
            if tau == G:
                K_actual = 1
                # Shape: (1, G) containing indices 0.
                subgroup_indices = torch.zeros(1, G, dtype=torch.long, device=device)
            else:
                K_actual = K
                # Create mask of shape (B, G, K_actual, G)
                rand_indices = torch.rand(B, G, K_actual, G, device=device).argsort(dim=-1)
                
                # Ensure that index `g` is always part of its own sampled subgroups!
                g_idx = torch.arange(G, device=device).view(1, G, 1, 1).expand(B, G, K_actual, 1)
                mask_neq_g = rand_indices != g_idx
                
                _, valid_idx_pos = torch.sort(mask_neq_g.long(), dim=-1, descending=True)
                other_indices = torch.gather(rand_indices, -1, valid_idx_pos)[..., :tau-1]
                
                # Combine `g` and the other random indices to form the subgroup!
                subgroup_indices = torch.cat([g_idx, other_indices], dim=-1) # (B, G, K_actual, tau)

            # 4. Extract Rewards for Subgroups
            if tau == G:
                # Full scale shortcut: (B, G, 1) -> (B, 1, 1, G, 1) -> (B, G, K=1, tau=G, T=1)
                sub_rewards = rewards.unsqueeze(1).unsqueeze(2).expand(B, G, 1, G, 1)
            else:
                # Expand rewards: (B, 1, 1, G, 1)
                rewards_expanded = rewards.view(B, 1, 1, G, -1).expand(B, G, K_actual, G, -1)
                # Expand indices: (B, G, K, tau, 1)
                indices_expanded = subgroup_indices.unsqueeze(-1).expand(-1, -1, -1, -1, 1)
                
                sub_rewards = torch.gather(rewards_expanded, 3, indices_expanded)

            # 5. Compute Advantage within the Subgroup
            r_i = sub_rewards[..., 0, :] # (B, G, K, 1)
            mu_S = sub_rewards.mean(dim=3) # (B, G, K, 1)
            
            # Use unbiased=False to prevent std blowup on tiny τ subsets
            sigma_S = sub_rewards.std(dim=3, unbiased=False) + self.epsilon # (B, G, K, 1)
            
            adv_S = (r_i - mu_S) / sigma_S # (B, G, K, 1)
            
            # Average across the K samples for this scale
            avg_adv_scale = adv_S.mean(dim=2) # (B, G, 1)
            all_scale_advantages.append(avg_adv_scale)

        # 6. Weighted Sum Across Scales
        all_scale_advantages = torch.stack(all_scale_advantages, dim=-1) # (B, G, 1, num_scales)
        weights_view = weights.view(1, 1, 1, -1)
        final_advantages = (all_scale_advantages * weights_view).sum(dim=-1) # (B, G, 1)

        return final_advantages.squeeze(-1) # (B, G)