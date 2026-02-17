"""
Priority Buffer — Entries ranked by a priority score.

═══════════════════════════════════════════════════════════════════
STATUS: DEFERRED — Needed only for advanced replay strategies.
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Stores BufferEntry objects sorted by a priority score.
    When sampling, returns highest-priority entries first.

POTENTIAL USE CASES:
    - Training on the "hardest" prompts first (priority = 1 - μ_q)
    - Prioritized experience replay (priority = TD-error analogue)
    - XRPO-style prioritization (but XRPO priority lives in the planner,
      not the buffer — so this may not be needed for XRPO at all)

IMPLEMENTATION:
    Backed by a heap or sorted list.
    insert() assigns priority score.
    get_top_k(K) returns K highest-priority entries.

FOR NOW:
    Use FIFOBuffer. Priority logic lives in the Planner (adaptive.py)
    or in the Coordinator's loop, not in the buffer.
"""
