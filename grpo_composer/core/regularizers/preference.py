"""
Preference Regularizers

This module implements preference-based regularization (AMIR-GRPO).
Uses intra-group reward rankings to form implicit preference pairs and apply DPO-style loss.

Paper: AMIR-GRPO
----------------
Adds a DPO-style contrastive preference regularizer using implicit preference pairs 
from intra-group reward rankings.

Formulas:
---------
1. Implicit Preference Set:
   S(q) = {(i, j) | r_i > r_j, r_i - r_j > δ_r}
   (Only pairs with sufficient reward gap are considered)

2. DPO-Style Logit:
   z_{i,j}(θ) = β_DPO * [(ℓ_θ(q, o_i) - ℓ_ref(q, o_i)) - (ℓ_θ(q, o_j) - ℓ_ref(q, o_j))]
   where ℓ is length-normalized log-probability.

3. Preference Regularizer:
   J_pref(θ) = E_{(q,i,j) ~ S} [ log σ(z_{i,j}(θ)) ]

4. Final Objective:
   J_AMIR-GRPO(θ) = J_GRPO(θ) + λ_reg * J_pref(θ)

Note on sign convention:
   J_pref is a REWARD (to maximize). Since we MINIMIZE loss in the composer,
   we return -J_pref so that: loss = pg_loss + λ_reg * (-J_pref)
   effectively does: loss = pg_loss - λ_reg * J_pref (maximizes J_pref).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .base import Regularizer

class PreferenceRegularizer(Regularizer):
    """
    AMIR-GRPO Preference Regularizer.
    
    Constructs implicit preference pairs from the generated group based on reward differences.
    Applies a DPO-style objective to these pairs to reinforce the preference for higher-reward outputs.
    """
    
    def __init__(
        self,
        beta_dpo: float = 0.1,  # Corresponds to β_DPO in formula
        delta_reward: float = 0.0, # Minimum reward gap for valid pair
    ):
        super().__init__()
        self.beta_dpo = beta_dpo
        self.delta_reward = delta_reward
        
    def compute_regularization(
        self,
        log_probs: torch.Tensor,        # (B, T) — current policy log-probs per token
        ref_log_probs: torch.Tensor,    # (B, T) — reference policy log-probs per token
        rewards: torch.Tensor,          # (B,) — scalar reward per sequence
        mask: torch.Tensor,             # (B, T) — valid token mask
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the AMIR-GRPO preference regularization term.
        
        Steps:
        1. Compute length-normalized sequence log-probs:
           ℓ_θ(o_i) = (1/|o_i|) Σ_t log π_θ(o_{i,t}|...)
           ℓ_ref(o_i) = (1/|o_i|) Σ_t log π_ref(o_{i,t}|...)
        
        2. Identify valid pairs (i, j) where r_i > r_j + δ_r
        
        3. For each pair, compute DPO logit:
           z_{i,j} = β_DPO * [(ℓ_θ(o_i) - ℓ_ref(o_i)) - (ℓ_θ(o_j) - ℓ_ref(o_j))]
        
        4. Compute log σ(z_{i,j}) for all pairs
        
        5. Return negative mean (we minimize loss, but J_pref should be maximized)

        Shape Flow:
            log_probs: (B, T), ref_log_probs: (B, T), mask: (B, T)
            seq_lengths: (B,) — count valid tokens
            ell_theta: (B,) — length-normalized log-prob under current policy
            ell_ref: (B,) — length-normalized log-prob under reference
            log_ratio: (B,) — ℓ_θ - ℓ_ref per sequence
            z_{i,j}: (num_pairs,) — DPO logits for all valid pairs
            loss: scalar — -mean(log σ(z))
        """
        B = log_probs.shape[0]
        
        # Step 1: Length-normalized sequence log-probs
        seq_lengths = mask.sum(dim=-1).clamp(min=1)  # (B,)
        ell_theta = (log_probs * mask).sum(dim=-1) / seq_lengths     # (B,)
        ell_ref = (ref_log_probs * mask).sum(dim=-1) / seq_lengths   # (B,)
        
        # Log-ratio per sequence: ℓ_θ(o_i) - ℓ_ref(o_i)
        log_ratio = ell_theta - ell_ref  # (B,)
        
        # Step 2: Construct implicit preference pairs
        # S(q) = {(i, j) | r_i > r_j, r_i - r_j > δ_r}
        # For all i, j in the batch (treating the batch as one group)
        logits = []
        for i in range(B):
            for j in range(B):
                if rewards[i] > rewards[j] and (rewards[i] - rewards[j]) > self.delta_reward:
                    # Step 3: DPO logit
                    # z_{i,j} = β_DPO * (log_ratio_i - log_ratio_j)
                    z_ij = self.beta_dpo * (log_ratio[i] - log_ratio[j])
                    logits.append(z_ij)
        
        if len(logits) == 0:
            # No valid pairs (all same reward), return zero
            return torch.tensor(0.0, device=log_probs.device)
        
        # Step 4: Compute log σ(z_{i,j})
        z = torch.stack(logits)  # (num_pairs,)
        log_sigmoid_z = F.logsigmoid(z)  # (num_pairs,)
        
        # Step 5: Return NEGATIVE mean (since we minimize loss, but want to maximize J_pref)
        # J_pref = mean(log σ(z)) — this is a reward (higher = better)
        # Loss contribution = -J_pref (so minimizing loss maximizes J_pref)
        return -log_sigmoid_z.mean()
