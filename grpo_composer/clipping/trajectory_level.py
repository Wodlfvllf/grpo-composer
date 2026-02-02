
"""
Trajectory-Level Clipping Mechanism (TIC-GRPO)

From: TIC-GRPO paper

Key Differences from Standard GRPO:
-----------------------------------
1. Standard GRPO: Token-level ratio π_θ(s_t|s_{t-1}) / π_θ_old(s_t|s_{t-1})
2. TIC-GRPO: Trajectory-level ratio P_θ(s_T|s_0) / P_θ_old(s_T|s_0)
3. Upper-only clipping: Only clips to prevent ratio from going too high

The trajectory-level ratio is computed as the product of token-level ratios,
or equivalently, exp(sum of log-ratios).

Input:
------
- log_probs: torch.Tensor, shape (B, G, T) - log probs of tokens under current policy
- ref_log_probs: torch.Tensor, shape (B, G, T) - log probs under reference policy
- attention_mask: torch.Tensor, shape (B, G, T) - mask for valid tokens

Output:
-------
- clipped_ratios: torch.Tensor, shape (B, G) - trajectory-level clipped ratios
"""

import torch
from .base import ClippingMechanism


class TrajectoryLevelClippingMechanism(ClippingMechanism):
    """
    TIC-GRPO: Upper-only clipping at trajectory level.
    
    Instead of clipping token-level ratios, we:
    1. Compute trajectory-level ratio (product of token ratios)
    2. Apply upper-only clipping (no lower bound)
    """
    
    def __init__(self, clip_upper: float = 1.28, clip_lower: float = None):
        """
        Args:
            clip_upper: Upper bound for trajectory ratio (default 1.28 from DAPO)
            clip_lower: Lower bound (None = no lower clipping, as per TIC-GRPO)
        """
        self.clip_upper = clip_upper
        self.clip_lower = clip_lower  # None means no lower clipping
    
    def compute_trajectory_ratio(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute trajectory-level probability ratio.
        
        Args:
            log_probs: (B, G, T) log probs under current policy
            ref_log_probs: (B, G, T) log probs under reference policy
            attention_mask: (B, G, T) mask for valid tokens
            
        Returns:
            trajectory_ratio: (B, G) ratio of trajectory probabilities
        """
        # Log-ratio per token
        log_ratio = log_probs - ref_log_probs  # (B, G, T)
        
        if attention_mask is not None:
            # Sum only over valid tokens
            log_ratio = log_ratio * attention_mask
            trajectory_log_ratio = log_ratio.sum(dim=-1)  # (B, G)
        else:
            trajectory_log_ratio = log_ratio.sum(dim=-1)  # (B, G)
        
        # Convert back to ratio space
        trajectory_ratio = torch.exp(trajectory_log_ratio)  # (B, G)
        
        return trajectory_ratio
    
    def clip(
        self,
        probs_ratio: torch.Tensor = None,
        log_probs: torch.Tensor = None,
        ref_log_probs: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply upper-only clipping at trajectory level.
        
        Can accept either:
        - probs_ratio: Pre-computed trajectory-level ratio (B, G)
        - log_probs + ref_log_probs: To compute trajectory ratio internally
        
        Returns:
            clipped_ratio: (B, G) trajectory-level clipped ratios
        """
        if probs_ratio is None:
            if log_probs is None or ref_log_probs is None:
                raise ValueError(
                    "Must provide either probs_ratio or (log_probs, ref_log_probs)"
                )
            probs_ratio = self.compute_trajectory_ratio(
                log_probs, ref_log_probs, attention_mask
            )
        
        # Upper-only clipping (TIC-GRPO characteristic)
        if self.clip_lower is not None:
            clipped = torch.clamp(probs_ratio, min=self.clip_lower, max=self.clip_upper)
        else:
            # Upper-only: no lower bound
            clipped = torch.clamp(probs_ratio, max=self.clip_upper)
        
        return clipped
