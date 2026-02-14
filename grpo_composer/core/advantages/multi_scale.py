"""
MS-GRPO: Multi-Scale Advantage via Hierarchical Subgroup Comparisons

Paper: MS-GRPO

Components Changed (from base GRPO):
- Advantage computed at MULTIPLE scales τ ∈ {τ_min, ..., G}
- Final advantage is weighted average across scales

Mathematical Form:
    Single-scale advantage for subgroup S:
        Â_{i,t}(S) = (r_{i,t} - μ_S) / σ_S

    Scale-specific average:
        Ā^(τ)_{i,t} = (1 / C(G-1, τ-1)) * Σ_{|S|=τ, o_i∈S} Â_{i,t}(S)

    Multi-scale advantage:
        Â^{MS-GRPO}_{i,t} = Σ_{τ=τ_min}^G w_τ * Ā^(τ)_{i,t}

    Weights: w_τ ≥ 0, Σ w_τ = 1 (default: uniform)
"""

import torch
from itertools import combinations
from .base import AdvantageFunction
class MultiScaleAdvantageFunction(AdvantageFunction):
    
    def __init__(self, tau_min: int = 2, tau_max: int = None, weights: list = None, epsilon: float = 1e-8):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max  # Will default to G if None
        self.weights = weights  # Will default to uniform if None
        self.epsilon = epsilon
        
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (B, G) rewards
        Returns:
            advantages: (B, G) multi-scale advantages
        """
        B, G = rewards.shape
        tau_max = self.tau_max if self.tau_max is not None else G
        tau_min = max(self.tau_min, 2)  # Need at least 2 for std
        
        # Default uniform weights
        num_scales = tau_max - tau_min + 1
        if self.weights is None:
            weights = torch.ones(num_scales) / num_scales
        else:
            weights = torch.tensor(self.weights)
        
        all_advantages = torch.zeros(B, G)
        
        for b in range(B):
            r = rewards[b]  # (G,)
            sample_advantages = torch.zeros(G)
            
            for i in range(G):
                # For sample i, compute advantage at each scale
                scale_advantages = []
                
                for scale_idx, tau in enumerate(range(tau_min, tau_max + 1)):
                    # Find all subgroups of size tau that include sample i
                    other_indices = [j for j in range(G) if j != i]
                    subgroup_advantages = []
                    
                    # Generate subgroups: {i} ∪ (τ-1 others)
                    for others in combinations(other_indices, tau - 1):
                        subgroup_idx = [i] + list(others)
                        subgroup_rewards = r[subgroup_idx]
                        
                        # Advantage within subgroup
                        mean = subgroup_rewards.mean()
                        std = subgroup_rewards.std() + self.epsilon
                        adv_i = (r[i] - mean) / std
                        subgroup_advantages.append(adv_i)
                    
                    # Average over subgroups at this scale
                    if subgroup_advantages:
                        scale_advantages.append(torch.stack(subgroup_advantages).mean())
                
                # Weighted sum across scales
                if scale_advantages:
                    sample_advantages[i] = (torch.stack(scale_advantages) * weights[:len(scale_advantages)]).sum()
            
            all_advantages[b] = sample_advantages
        
        return all_advantages