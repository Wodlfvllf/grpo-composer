"""
DARO Batch Sampler (Difficulty-Aware Rejection Sampling Optimization)

Implements the exact DARO sampling strategy from the paper.

Paper: DARO - Difficulty-Aware Rejection Sampling Optimization
Algorithm: Group prompts by difficulty bins with per-group token normalization

Purpose:
-------
Sample prompts grouped by difficulty level μ (mean accuracy over G completions).
Uses per-group normalization to ensure equal contribution from each difficulty bin.

Role in Training Pipeline:
-------------------------
**Config**: `sampler: daro`
**Phase 4**: Batch Sampling
**Goal**: Balanced learning across difficulty spectrum

Key Algorithm (from paper):
---------------------------
1. **Difficulty Bins**: M = {1/G, 2/G, ..., (G-1)/G}
   - μq = mean accuracy of G completions for prompt q
   - Exclude μ = 0 (all incorrect) and μ = 1 (all correct) → zero gradient
   
2. **Per-Group Normalization**: Ω_μ = 1 / L_μ
   - L_μ = total tokens in difficulty group μ
   - Normalizes contribution of each group

3. **Sampling**: 
   - Filter out μ ∈ {0, 1}
   - Group remaining prompts by μ
   - Sample from each group with per-group weight

Mathematical Formulation:
------------------------
```
Difficulty bins:
M = {1/G, 2/G, 3/G, ..., (G-1)/G}

For each prompt q:
μ_q = (1/G) Σ_{i=1}^G correct_i

Filtering:
Keep only prompts where μ_q ∈ M (i.e., 0 < μ_q < 1)

Per-group normalization:
Ω_μ = 1 / L_μ
where L_μ = Σ_{q: μ_q=μ} |completion_tokens|_q

Loss contribution:
Each group contributes equally after normalization
```

Example Usage:
-------------
```python
# Create DARO sampler
sampler = DAROBatchSampler(
    group_size=8,
    num_bins=10,
    binning_strategy="quantile",
    update_weights=True  # Learn bin weights
)

# Sample batch
batch = sampler.sample(buffer, batch_size=32)

# batch.difficulties contains difficulty scores for monitoring
```

Integration with LogWeightRegularizer:
--------------------------------------
```python
# In training loop
batch = sampler.sample(buffer, batch_size)

# Compute loss (includes log_weight regularizer)
loss = loss_function.compute_loss(...)

# Regularizer may update bin weights
if isinstance(sampler, DAROBatchSampler):
    sampler.update_bin_weights(regularizer.get_bin_weights())
```

Implementation Notes:
--------------------
- Recompute bin boundaries periodically as difficulty distribution shifts
- Log bin statistics for monitoring (count per bin, avg reward, etc.)
- Handle empty bins gracefully
- Support mixed sampling if bin is depleted
- Compatible with LogWeightRegularizer for learnable weights

Edge Cases:
----------
- Bin has zero prompts → skip or sample from neighbors
- All prompts same difficulty → fallback to uniform
- Batch size not divisible by num_bins → round and adjust
- Buffer smaller than batch_size → sample with replacement
"""
