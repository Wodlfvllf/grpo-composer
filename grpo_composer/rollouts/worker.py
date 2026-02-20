"""
Rollout Worker — Generates G completions per prompt and packs into BufferEntry.

═══════════════════════════════════════════════════════════════════
INPUT (from DataLoader via Coordinator):
═══════════════════════════════════════════════════════════════════

    The DataLoader + DataCollator produce a batch dict:
        input_ids      : (B, T)      — tokenized prompt+target, padded
        attention_mask : (B, T)      — 1=real, 0=padding
        labels         : (B, T)      — same as input_ids, padding masked with -100

    The Worker receives input_ids and attention_mask.
    Each row i in input_ids is ONE prompt (B prompts total).

═══════════════════════════════════════════════════════════════════
WHAT THE WORKER DOES (per prompt):
═══════════════════════════════════════════════════════════════════

    For each prompt i in the batch (i = 0..B-1):

    1. Extract prompt tokens:  input_ids[i]  → (T,)
    
    2. Generate G completions:
       Generator.generate(prompt, G) → G completions
       Each completion j has:
           completion_tokens[j]  : (T_completion_j,)   ← variable length
           policy_log_probs[j]   : (T_completion_j,)   ← log π_θold(token | context)

    3. Score each completion:
       RewardEvaluator.compute_rewards(prompt, completions) → (G,) rewards

    4. Get reference log-probs (for KL and ratio):
       ReferenceModel.get_log_probs(completion_tokens) → (G, T_completion) ref_log_probs

    5. Pack into BufferEntry:
       BufferEntry(
           completions        = (G, T_completion),   ← padded to max within group
           policy_log_probs   = (G, T_completion),
           ref_log_probs      = (G, T_completion),
           rewards            = (G,),
           completion_lengths = (G,),                 ← actual length before padding
           mean_accuracy      = float,                ← μ_q = mean(correctness)
           metadata           = {},
       )

═══════════════════════════════════════════════════════════════════
OUTPUT:
═══════════════════════════════════════════════════════════════════

    List[BufferEntry] — one per prompt in the batch (length B).
    
    These go into the FIFOBuffer, then the RolloutCollator stacks
    B entries into (B, G, T) tensors for the loss function.

═══════════════════════════════════════════════════════════════════
SHAPE FLOW:
═══════════════════════════════════════════════════════════════════

    DataLoader        →  {input_ids: (B, T), attention_mask: (B, T)}
                              │
    Worker (per prompt i):    │
        input_ids[i]          │  (T,)
            │                 │
            ▼                 │
        Generator.generate()  │  → G completions, each (T_comp_j,)
            │                 │
            ▼                 │
        pad within group      │  → (G, T_comp_max)   T_comp_max = max over G
            │                 │
            ▼                 │
        BufferEntry           │  completions: (G, T_comp_max)
                              │  policy_log_probs: (G, T_comp_max)
                              │  ref_log_probs: (G, T_comp_max)
                              │  rewards: (G,)
                              │
    RolloutCollator           │
        stacks B entries      │  → (B, G, T)     T = max_completion_length
                              ▼
    Loss receives (B, G, T) tensors

═══════════════════════════════════════════════════════════════════
VARIANT-SPECIFIC NOTES:
═══════════════════════════════════════════════════════════════════

    Most variants don't change the Worker. The Worker always produces
    the same BufferEntry structure. What changes per variant:

    - Which Generator is used (HF, vLLM, TRT-LLM)
    - Which RewardEvaluator is used (binary, frequency-aware, length-dependent)
    - Which ReferenceModel is used (frozen, EMA, or None if β=0)

    Exceptions where Worker needs extra metadata:
    - Unlikeliness-GRPO: store rank(y_i) by probability in metadata
    - GAPO: compute group frequency f_v(o) in metadata
    - DRA-GRPO: compute pairwise cosine sim in metadata
    - XRPO: novelty = mean(policy_log_probs), already available

═══════════════════════════════════════════════════════════════════
CONSTRUCTOR ARGS:
═══════════════════════════════════════════════════════════════════

    generator   : Generator         — from interfaces/generator.py
    reward_eval : RewardEvaluator   — from interfaces/reward_model.py
    ref_model   : ReferenceModel    — from interfaces/reference_model.py (optional)
    group_size  : int               — G (number of completions per prompt)
    tokenizer   : Tokenizer         — for decoding completions to text (reward needs text)
"""
import torch
import torch.nn as nn
from typing import List
from ..interfaces import (
    InferenceEngine, 
    RewardEvaluator, 
    ReferenceModel, 
    BufferEntry, 
    RolloutResult, 
    RolloutRequest
)
from .collator import RolloutCollator
def expand_for_group(input_ids, attention_mask, G):
    """(B, T) → (B*G, T) by repeating each prompt G times."""
    return input_ids.repeat_interleave(G, dim=0), attention_mask.repeat_interleave(G, dim=0)


class Worker:
    def __init__(
        self,
        policy_generator: InferenceEngine,
        policy_evaluator : 
        reward_evaluator: RewardEvaluator,
        reference_evaluator: InferenceEngine,
        group_size: int = 8,
        max_completion_length: int = 512,
        pad_value: float = -100.0,
    ):
        self.generator = generator
        self.reward_evaluator = reward_evaluator
        self.reference_model = reference_model
        self.group_size = group_size
        self.max_completion_length = max_completion_length
        self.pad_value = pad_value
        self.collator = RolloutCollator(self.max_completion_length, self.pad_value)

    def generate_batch(self, batch):
        """
        batch: dict from DataLoader
            input_ids:      (B, T)
            attention_mask: (B, T)

        Returns: dict from RolloutCollator
            log_probs:      (B, G, T_comp)
            ref_log_probs:  (B, G, T_comp)
            rewards:        (B, G)
            mask:           (B, G, T_comp)
        """
        input_ids = batch["input_ids"]            # (B, T)
        attention_mask = batch["attention_mask"]   # (B, T)

        B = input_ids.size(0)
        G = self.group_size

        # 1. Expand prompts: (B, T) → (B*G, T)
        expanded_ids, expanded_mask = expand_for_group(input_ids, attention_mask, G)

        request = RolloutRequest(
            input_ids = expanded_ids,
            attention_mask = expanded_mask
        )
        # 2. Generate all completions at once
        rollout = self.generator.generate(
            request = request
        )
        # rollout.completions: (B*G, L)
        # rollout.log_probs:   (B*G, L)

        # 3. Reshape to (B, G, L)
        L = rollout.completions.size(-1)
        completions = rollout.completions.view(B, G, L)
        policy_log_probs = rollout.log_probs.view(B, G, L)

        # 4. Rewards: (B, G)
        rewards = self.reward_evaluator(input_ids, completions)

        # 5. Reference log-probs: (B*G, L) → (B, G, L)
        ref_log_probs = self.reference_model.get_log_probs(
            rollout.completions    # (B*G, L) — ref model scores completions
        ).view(B, G, L)

        # 6. Pack one BufferEntry per prompt
        entries = []
        for i in range(B):
            entries.append(BufferEntry(
                completions=completions[i],           # (G, L)
                policy_log_probs=policy_log_probs[i], # (G, L)
                ref_log_probs=ref_log_probs[i],       # (G, L)
                rewards=rewards[i],                   # (G,)
            ))

        # 7. Collate → (B, G, T) padded tensors
        return self.collator.collate(entries)
