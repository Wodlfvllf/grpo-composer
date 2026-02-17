"""
Fixed Planner — Sequential iteration over the dataset.

The simplest planner. Iterates through the dataset in order, returning the
next N prompts each time select() is called. Wraps around at epoch boundary.

═══════════════════════════════════════════════════════════════════
USED BY:
    Standard GRPO, Dr.GRPO, λ-GRPO, DAPO, DARO, KRPO, GAPO,
    DRA-GRPO, GDPO, GRPO-LEAD, MS-GRPO, P-GRPO, TIC-GRPO, TR-GRPO,
    Stratified-GRPO, Unlikeliness-GRPO, SPO, AMIR-GRPO, Info-GRPO
    
    Basically everything EXCEPT XRPO (which needs adaptive priority).
═══════════════════════════════════════════════════════════════════

BEHAVIOR:
    Call 1: select(dataset, N=32) → prompts[0:32]
    Call 2: select(dataset, N=32) → prompts[32:64]
    Call 3: select(dataset, N=32) → prompts[64:96]
    ...
    Call K: select(dataset, N=32) → wraps around to start (new epoch)

    Maintains an internal cursor: self._cursor
    
    Optional: shuffle dataset at each epoch boundary for randomization.

═══════════════════════════════════════════════════════════════════
CONSTRUCTOR ARGS:
═══════════════════════════════════════════════════════════════════

    shuffle : bool = True    — shuffle at epoch boundary
    seed    : int = 42       — for reproducibility
"""
