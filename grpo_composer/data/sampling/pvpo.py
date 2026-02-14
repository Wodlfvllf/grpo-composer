"""
PVPO Batch Sampler (Process-supervised Value-guided Policy Optimization)

Implements PVPO filtering + Ground Truth trajectory injection.

Paper: PVPO - Filtering with GT trajectory injection for zero-accuracy samples
Algorithm: Filter trivial prompts + inject GT for failed prompts

Purpose:
-------
1. Filter prompts where all completions are correct (trivial, μ=1)
2. Retain prompts with mixed quality (0 < μ < 1)
3. For prompts where all fail (μ=0), inject GT trajectory from stronger LLM

Role in Training Pipeline:
-------------------------
**Config**: `sampler: pvpo`
**Phase 4**: Batch Sampling
**Special**: Requires GT trajectory cache for μ=0 prompts

Key Algorithm (from paper):
---------------------------
**Filtering**:
- Exclude: μ_q = 1 (all correct, trivial)  
- Retain: 0 < μ_q < 1 (nonzero advantage)
- Special: μ_q = 0 (all incorrect) → use GT injection

**GT Injection for μ=0**:
1. Use larger/stronger LLM to generate correct trajectory
2. Cache GT trajectory for prompt
3. Replace one of the G rollouts with cached GT
4. Result: μ becomes 1/G (one correct out of G)
5. Now 0 < μ < 1 → usable for training

**Rationale**:
- μ = 1: No room for improvement, waste of compute
- 0 < μ < 1: Useful signal (some good, some bad completions)
- μ = 0: Without GT, zero gradient. With GT, learn from contrast

Mathematical Formulation:
------------------------
```
For each prompt q:
μ_q = (1/G) Σ_{i=1}^G correct_i

Filtering rules:
- If μ_q = 1: Exclude (trivial)
- If 0 < μ_q < 1: Include as-is
- If μ_q = 0: 
    if GT_cached(q):
        Replace rollout_0 with GT_trajectory
        μ_q ← 1/G
        Include
    else:
        Query stronger LLM for GT
        Cache GT
        Replace and include
```


Example Usage:
-------------
```python
# Create PVPO sampler with GT generator
from grpo_composer.inference.engines import HFGenerator

stronger_llm = HFGenerator(model_name="gpt-4")  # Or larger model

sampler = PVPOBatchSampler(
    group_size=8,
    gt_cache={},  # Will populate over time
    gt_generator=stronger_llm,
    inject_position=0  # Replace first rollout
)

# Sample batch
batch = sampler.sample(buffer, batch_size=32)
# Prompts with μ=1 filtered
# Prompts with μ=0 have GT injected
```

GT Cache Management:
-------------------
```python
# Save cache to disk
import json
with open("gt_cache.json", "w") as f:
    json.dump(sampler.gt_cache, f)

# Load cache
with open("gt_cache.json", "r") as f:
    gt_cache = json.load(f)

sampler = PVPOBatchSampler(gt_cache=gt_cache, ...)
```

Implementation Notes:
--------------------
- GT generation can be expensive - cache aggressively
- Stronger LLM should be significantly better (GPT-4, Claude, larger model)
- inject_position can be randomized to avoid bias
- Monitor GT cache hit rate for efficiency
- Consider pre-generating GTs offline for common datasets

Comparison with Other Samplers:
-------------------------------
| Sampler | Filter μ=1 | Filter μ=0 | GT Injection |
|---------|------------|------------|--------------|
| DAPO    | Yes        | Yes        | No           |
| DARO    | Yes        | Yes        | No           |
| PVPO    | Yes        | No*        | Yes*         |

*PVPO doesn't filter μ=0, it injects GT instead

Edge Cases:
----------
- No GT generator provided → fallback to filtering μ=0
- GT generation fails → skip prompt or retry
- GT cache corrupted → regenerate
- All prompts μ=1 → raise error, dataset too easy
"""
