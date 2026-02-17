"""
rollouts — The generation-side pipeline.

Produces TrainingBatch(B, G, T) tensors that the loss function consumes.

Architecture:
─────────────
    Trainer calls ONLY → coordinator.step()
    
    Coordinator orchestrates:
        planner.select()   → which prompts to generate for
        worker.generate()  → G completions + rewards + log_probs per prompt
        collator.collate() → stack into (B, G, T) tensors
    
    Coordinator handles variant-specific loops:
        Simple      : one-shot generation    (GRPO, Dr.GRPO, λ-GRPO, DARO, ...)
        Oversampling: loop until 0<μ<1       (DAPO)
        GT Injection: inject GT for μ=0      (PVPO)
        Priority    : UCB prompt selection   (XRPO)

Module Map:
───────────
    coordinator.py       — Central orchestrator (entry point)
    worker.py            — Generate + score → BufferEntry
    collator.py          — List[BufferEntry] → TrainingBatch(B,G,T)
    scheduler.py         — When to generate (deferred, merge into coordinator)
    storage.py           — Persistent save/load (deferred)
    
    planner/
        base.py          — Planner ABC
        fixed.py         — Sequential iteration (most variants)
        adaptive.py      — UCB priority (XRPO)
        curriculum.py    — Difficulty progression (deferred)
    
    queue/
        sync.py          — Blocking generation (default)
        async_.py        — Non-blocking with polling (deferred, for Ray)

Implementation Priority:
────────────────────────
    1. worker.py + collator.py + planner/fixed.py    (core pipeline)
    2. coordinator.py with simple pattern             (get training working)
    3. coordinator.py oversampling pattern             (DAPO)
    4. planner/adaptive.py                            (XRPO)
    5. Everything else                                (scale features)
"""
