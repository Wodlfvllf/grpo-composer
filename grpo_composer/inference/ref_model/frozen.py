"""
Frozen Reference Model (Local)

In-memory frozen copy of policy model for computing reference log-probs.

What it does:
------------
- Loads model into memory (separate GPU or CPU)
- Freezes all parameters (no gradients)
- Computes log-probs via direct forward pass

When to use:
-----------
- Small models (≤13B) that fit in memory
- Single node training
- Simple setup needed

Example:
-------
```python
frozen_ref = FrozenReferenceModel(
    model=ref_model_copy,
    device="cuda:4"  # Separate GPU from policy
)

ref_log_probs = frozen_ref.get_log_probs(token_ids, mask)
```
"""
