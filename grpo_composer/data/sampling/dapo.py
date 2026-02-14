"""
DAPO Batch Sampler (Dynamic Sampling / Oversampling)

Implements the DAPO filtering strategy from the paper.

Paper: DAPO - Dynamic Sampling Policy Optimization
Algorithm: Filter uninformative prompts (μ=0 or μ=1)

Purpose:
-------
Dynamically filter out prompts that provide no gradient signal.
Only train on prompts where mean accuracy 0 < μ < 1.

Role in Training Pipeline:
-------------------------
**Config**: `sampler: dapo`
**Phase 4**: Batch Sampling
**Goal**: Skip zero-variance prompts to improve sample efficiency

Key Algorithm (from paper):
---------------------------
**Filter Function**: IOS(q) = I[0 < μ_q < 1]
- I[·] is indicator function (1 if true, 0 if false)
- μ_q = mean accuracy over G completions for prompt q

**Rationale**:
- μ = 1: All responses correct → advantages all zero → no gradient
- μ = 0: All responses incorrect → advantages all zero → no gradient
- 0 < μ < 1: Mixed quality → nonzero advantages → useful gradient

**Action**:
Skip prompts where IOS(q) = 0 (i.e., μ ∈ {0, 1})

Mathematical Formulation:
------------------------
```
For each prompt q:
μ_q = (1/G) Σ_{i=1}^G correct_i

Informativeness:
IOS(q) = I[0 < μ_q < 1]

Sampling set:
S = {q : IOS(q) = 1}

Sample batch uniformly from S
```


Example Usage:
-------------
```python
# Create DAPO sampler
sampler = DAPOBatchSampler(
    group_size=8,
    min_valid_ratio=0.1
)

# Sample batch - only informative prompts
batch = sampler.sample(buffer, batch_size=32)

# All prompts in batch have 0 < μ < 1
```

Comparison with DARO:
--------------------
| Aspect | DAPO | DARO |
|--------|------|------|
| **Filtering** | Yes (μ ∈ {0,1}) | Yes (μ ∈ {0,1}) |
| **Grouping** | No | Yes (by difficulty bins) |
| **Weighting** | Uniform | Per-group normalization Ω_μ |
| **Use Case** | Simple filtering | Balanced difficulty learning |

When to Use DAPO:
-----------------
- Simple baseline that filters uninformative samples
- When you don't need per-difficulty balancing
- When computational efficiency is important (no grouping overhead)

Implementation Notes:
--------------------
- Requires "correct" field in rollout metadata
- Monitor filter rate - if >90% filtered, dataset may be too easy/hard
- Can combine with other strategies (e.g., DAPO + prioritized replay)
- Efficient: O(N) filtering where N = number of prompts

Edge Cases:
----------
- All prompts filtered → raise error, suggest dataset/rollout changes
- Very few valid prompts → log warning, may need more rollouts per prompt
- Batch size > valid prompts → raise error

Oversampling Variant (future):
------------------------------
Could extend to oversample near-boundary difficulties (μ ≈ 0.5):
```python
weight[q] = 1 - 4(μ_q - 0.5)²  # Peaked at μ=0.5
```
This emphasizes "medium difficulty" prompts.
"""
