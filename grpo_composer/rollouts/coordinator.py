"""
Rollout Coordinator — THE central orchestrator of the rollout pipeline.

This is the most important file in rollouts/. It owns the entire flow:
    Planner → Worker → Filter/Loop → Collate → TrainingBatch

The Trainer calls ONLY coordinator.step() — nothing else.

═══════════════════════════════════════════════════════════════════
WHO CALLS THIS:
    training/trainer.py  →  coordinator.step(dataset, batch_size, step)
    
WHAT THIS CALLS:
    planner.select()     →  which prompts to generate for
    worker.generate()    →  G completions + rewards + log_probs per prompt
    collator.collate()   →  stack List[BufferEntry] → TrainingBatch(B,G,T)
═══════════════════════════════════════════════════════════════════

WHAT THIS RETURNS:
    TrainingBatch with fixed shapes:
        log_probs      :  (B, G, T)   — current policy log-probs
        ref_log_probs  :  (B, G, T)   — reference model log-probs (for KL + ratio)
        rewards        :  (B, G)      — per-completion scalar rewards
        mask           :  (B, G, T)   — 1=valid token, 0=padding

═══════════════════════════════════════════════════════════════════
LOOP PATTERNS (variant-specific):
═══════════════════════════════════════════════════════════════════

Pattern A — Simple (GRPO, Dr.GRPO, λ-GRPO, DARO, KRPO, etc.):
    prompts = planner.select(dataset, batch_size)
    entries = worker.generate(prompts)
    return collator.collate(entries)
    
    One shot. No loop. Worker called once.

Pattern B — Oversampling (DAPO):
    valid = []
    while len(valid) < batch_size:
        prompts = planner.select(dataset, needed)
        entries = worker.generate(prompts)
        valid.extend([e for e in entries if 0 < e.mean_accuracy < 1])
    return collator.collate(valid[:batch_size])
    
    LOOPS until enough valid (0 < μ < 1) entries.
    Worker called MULTIPLE times.
    Filter condition: discard μ=0 (all wrong) and μ=1 (all correct).

Pattern C — GT Injection (PVPO):
    prompts = planner.select(dataset, batch_size)
    entries = worker.generate(prompts)
    for e in entries:
        if e.mean_accuracy == 0:
            gt = gt_generator.generate(e.prompt)   ← STRONGER LLM
            e.completions[0] = gt
            e.rewards[0] = 1.0
        elif e.mean_accuracy == 1:
            discard e
    return collator.collate(valid[:batch_size])
    
    Worker called once. GT generator called for μ=0 prompts.
    Requires a separate stronger model for ground truth.

Pattern D — Priority (XRPO):
    scores = planner.compute_priority(dataset, history, step)
    prompts = top-K by Π_q = Δ̂_q + φ_q
    entries = worker.generate(prompts)
    for e in entries where μ=0:
        augment prompt with ICL few-shot from success corpus
        re-entries = worker.generate(augmented_prompts)
    history.update(entries)
    return collator.collate(entries[:batch_size])
    
    Uses historical per-prompt stats for UCB priority.
    Worker may be called twice (normal + ICL-seeded).
    Planner is stateful (tracks n_q, s_q per prompt).

═══════════════════════════════════════════════════════════════════
INTERNAL STATE:
═══════════════════════════════════════════════════════════════════
    
    For simple/DAPO: no persistent state needed.
    For PVPO: self._gt_cache = {prompt_id: gt_trajectory}
    For XRPO: self._history = {prompt_id: {n_q, mean_reward, std_reward}}

═══════════════════════════════════════════════════════════════════
CONFIG FIELDS THAT CONTROL THIS:
═══════════════════════════════════════════════════════════════════

    config.rollout_strategy:  "simple" | "oversampling" | "gt_injection" | "priority"
    config.group_size:        G (completions per prompt)
    config.max_oversample_rounds:  safety limit for DAPO loop (default: 5)
    config.gt_generator:      optional Generator for PVPO GT injection
    config.icl_corpus:        optional dict for XRPO ICL seeding
"""

import torch
import torch.nn as nn

class Coordinator(nn.Module):
    def __init__(
            self,
            config
        ):
        """
        Config should contains - 
        1. Rollout Strategy or oversample strategy
        2. And what its parameters should be. Let us figure out later that we need to pass in the params as arguments of in config itself.
        """
        self.config = config

    def step(self):
        if config.rollout_strategy == "simple":
            

