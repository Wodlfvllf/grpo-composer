"""
Rollout Collator — Shapes List[BufferEntry] into TrainingBatch(B, G, T).

This is a simple utility. It takes the Worker's output (a list of BufferEntry
objects with variable-length completions) and stacks them into padded tensors
with a consistent (B, G, T) shape that the loss function expects.

═══════════════════════════════════════════════════════════════════
WHO CALLS THIS:
    rollouts/coordinator.py  →  collator.collate(entries)
    
WHAT THIS RETURNS:
    TrainingBatch — a dataclass with:
        log_probs      :  (B, G, T)   — padded, from BufferEntry.policy_log_probs
        ref_log_probs  :  (B, G, T)   — padded, from BufferEntry.ref_log_probs
        rewards        :  (B, G)      — stacked, from BufferEntry.rewards
        mask           :  (B, G, T)   — 1 where real token, 0 where padding
═══════════════════════════════════════════════════════════════════

WHY THIS EXISTS (instead of inline in coordinator):
    - Padding logic is non-trivial: completions have different lengths
    - T = max completion length across the batch (or a configured max_seq_len)
    - Shorter completions are right-padded with 0s
    - mask ensures padded positions don't contribute to loss

═══════════════════════════════════════════════════════════════════
PADDING STRATEGY:
═══════════════════════════════════════════════════════════════════

    For each BufferEntry:
        completion_i has length L_i (may differ per completion)
    
    T = max(L_i for all i across all entries in batch)
    
    For entry with completion length L_i < T:
        log_probs[..., L_i:] = 0.0
        ref_log_probs[..., L_i:] = 0.0  
        mask[..., L_i:] = 0              ← masks out padding

    This is identical to standard HuggingFace collation with padding.

═══════════════════════════════════════════════════════════════════
OPTIONAL EXTRAS IN TrainingBatch:
═══════════════════════════════════════════════════════════════════

    Depending on variant, TrainingBatch may also carry:
        completion_lengths : (B, G)      — for TIC-GRPO length normalization
        difficulty_scores  : (B,)        — for GRPO-LEAD difficulty-aware advantage
        metadata           : List[dict]  — for variant-specific data
    
    These are passed through as **kwargs to loss function.
"""

import torch
import torch.nn as nn 
from .interfaces import BufferEntry
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import List


class RolloutCollator:
    def __init__(self, max_completion_length: int = 512, pad_value: float = -100.0):
        self.max_completion_length = max_completion_length
        self.pad_value = pad_value

    def collate(self, batch: List[BufferEntry]):

        batch_policy = []
        batch_ref = []
        batch_rewards = []
        batch_mask = []

        T = self.max_completion_length

        for entry in batch:

            policy = entry.policy_log_probs        # (G, L)
            ref = entry.ref_log_probs              # (G, L)
            rewards = entry.rewards                # (G,)

            G, L = policy.shape

            # 1. Truncate if longer than T
            if L > T:
                policy = policy[:, :T]
                ref = ref[:, :T]
                L = T

            # 2. Pad if shorter than T
            pad_amount = T - L

            if pad_amount > 0:
                policy = F.pad(policy, (0, pad_amount), value=self.pad_value)
                ref = F.pad(ref, (0, pad_amount), value=self.pad_value)

                mask = torch.cat(
                    [
                        torch.ones(G, L, device=policy.device),
                        torch.zeros(G, pad_amount, device=policy.device),
                    ],
                    dim=1,
                )
            else:
                mask = torch.ones(G, T, device=policy.device)

            batch_policy.append(policy)
            batch_ref.append(ref)
            batch_rewards.append(rewards)
            batch_mask.append(mask)

        # Stack across batch
        batch_policy = torch.stack(batch_policy)     # (B, G, T)
        batch_ref = torch.stack(batch_ref)           # (B, G, T)
        batch_rewards = torch.stack(batch_rewards)   # (B, G)
        batch_mask = torch.stack(batch_mask)         # (B, G, T)

        return {
            "log_probs": batch_policy,
            "ref_log_probs": batch_ref,
            "rewards": batch_rewards,
            "mask": batch_mask,
        }




            
