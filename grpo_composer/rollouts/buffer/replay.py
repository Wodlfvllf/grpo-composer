"""
Replay Buffer — Persistent storage for off-policy rollout reuse.

═══════════════════════════════════════════════════════════════════
USED BY: PVPO (off-policy reuse of past rollouts)
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Unlike FIFOBuffer (cleared each step), ReplayBuffer persists across
    training steps. Old rollouts can be reused for multiple gradient updates.

WHY PVPO NEEDS THIS:
    PVPO uses a STATIC value baseline from the reference policy:
        Â_PVPO = r_i - mean(r_ref)
    
    The "r_ref" values come from past rollouts of the reference policy.
    These don't change between ref model updates, so they can be cached
    and reused. The ReplayBuffer stores them.

BEHAVIOR:
    insert(entries)                →  add new entries (does NOT clear old ones)
    sample(batch_size)             →  random sample from all stored entries
    evict_oldest(fraction=0.25)    →  remove oldest 25% when buffer is full
    size()                         →  total entries stored
    
    NOT cleared after each training step (unlike FIFOBuffer).

STALENESS:
    Old rollouts become stale as the policy θ diverges from θ_old.
    The ratio ρ = π_θ / π_θ_old becomes unreliable.
    
    Options:
    - Max age: evict entries older than K steps
    - Importance sampling correction: weight old entries by staleness
    - Periodic full refresh: re-generate all rollouts every N steps

CAPACITY:
    max_size     : int   — maximum number of entries
    evict_policy : str   — "oldest" | "lowest_reward" | "random"

═══════════════════════════════════════════════════════════════════
ALSO USEFUL FOR:
    - Pre-computed rollout datasets (generate offline, train online)
    - Mixed on-policy/off-policy training (freshness fraction)
"""
