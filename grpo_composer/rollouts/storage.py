"""
Rollout Storage — Persistent save/load of rollouts to disk.

═══════════════════════════════════════════════════════════════════
STATUS: DEFERRED — Not needed for initial implementation.
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Serialize BufferEntry objects to disk for:
    - Checkpointing (resume training from saved rollouts)
    - Offline analysis (inspect generated completions)
    - Pre-computed rollout datasets (generate once, train many times)

WHEN NEEDED:
    - When training crashes and you don't want to re-generate rollouts
    - When pre-generating rollouts on GPU cluster, then training on different hardware
    - For PVPO GT cache persistence

FORMAT OPTIONS:
    - torch.save() / torch.load() for simplicity
    - Apache Arrow / Parquet for large-scale datasets
    - HuggingFace datasets format for interoperability

FOR NOW:
    Skip. Checkpointing can be handled by training/callbacks/.
    Implement when you need persistent rollout storage.
"""
