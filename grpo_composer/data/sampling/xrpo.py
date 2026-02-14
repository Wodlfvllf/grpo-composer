"""
XRPO Batch Sampler (Exploration Policy Optimization)

Implements XRPO hierarchical rollout planning with exploration-exploitation.

Paper: XRPO - Hierarchical rollout planning with ICL seeding
Algorithm: UCB-style priority scoring with uncertainty reduction

Purpose:
-------
Prioritize prompts based on:
1. Uncertainty about prompt difficulty (exploration)
2. Expected reward improvement (exploitation)
3. ICL (in-context learning) seeding for zero-reward prompts

Role in Training Pipeline:
-------------------------
**Config**: `sampler: xrpo`
**Phase 4**: Batch Sampling
**Goal**: Balance exploration of uncertain prompts with exploitation of high-reward

Key Algorithm (from paper):
---------------------------
**Uncertainty (Confidence Interval)**:
h_q(n_q) = t_{1-α/2, n_q-1} * (s_q / √n_q)

Where:
- t_{1-α/2, n_q-1}: Student's t critical value
- s_q: Sample std dev of rewards for prompt q  
- n_q: Number of rollouts for prompt q
- α: Significance level (typically 0.05)

**Uncertainty Reduction**:
Δ̂_q(n_q) = h_q(n_q) - h_q(n_q + 1)
= Expected reduction in uncertainty from one more rollout

**Exploration Bonus**:
φ_q(T, n_q) = λ * log(1 + T) / n_q

Where:
- T: Total training steps so far
- λ: Exploration coefficient
- Encourages trying under-sampled prompts

**Priority Score**:
Π_q = Δ̂_q(n_q) + φ_q(T, n_q)

**ICL Seeding**:
For prompts with all rewards = 0, inject few-shot examples from success corpus

Mathematical Formulation:
------------------------
```
Per prompt q with n_q rollouts:

1. Compute sample statistics:
   μ_q = mean(rewards)
   s_q = std(rewards)

2. Uncertainty (confidence interval half-width):
   h_q(n_q) = t_{α/2}(n_q-1) * s_q/√n_q

3. Uncertainty reduction from one more sample:
   Δ̂_q(n_q) = h_q(n_q) - h_q(n_q+1)
             ≈ h_q(n_q) * (1 - √(n_q/(n_q+1)))

4. Exploration bonus (UCB-style):
   φ_q(T, n_q) = λ * log(1 + T) / n_q

5. Total priority:
   Π_q = Δ̂_q(n_q) + φ_q(T, n_q)

6. Sample proportional to Π_q
```

Example Usage:
-------------
```python
# Create XRPO sampler with success corpus
icl_corpus = {
    "math": [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Solve x+5=10", "answer": "x=5"},
    ],
    "code": [
        {"question": "Write fizzbuzz", "answer": "for i in range(1,101)..."},
    ]
}

sampler = XRPOBatchSampler(
    group_size=8,
    alpha=0.05,
   lambda_explore=0.1,
    icl_corpus=icl_corpus,
    use_icl_seeding=True
)

# Sample batch - high uncertainty + exploration bonus prompts prioritized
batch = sampler.sample(buffer, batch_size=32)

# Update step counter in training loop
sampler.update_training_step(trainer.global_step)
```

Priority Score Components:
--------------------------
```
For a prompt with:
- n_q = 10 rollouts
- s_q = 0.5 (reward std dev)
- T = 1000 (training steps)
- λ = 0.1

Uncertainty reduction:
Δ̂_q ≈ 0.03 (depends on t-distribution)

Exploration bonus:
φ_q = 0.1 * log(1001) / 10 ≈ 0.069

Total priority:
Π_q = 0.03 + 0.069 = 0.099
```

Interpretation:
---------------
- **High Δ̂_q**: Uncertain prompts (high variance, few samples)
- **High φ_q**: Under-explored prompts (low n_q, high T)
- **Balance**: XRPO naturally balances exploration-exploitation

Comparison with Other Samplers:
-------------------------------
| Sampler | Strategy | Prioritization |
|---------|----------|----------------|
| Uniform | Random | None |
| DAPO    | Filter | Informative only |
| DARO    | Stratify | Difficulty bins |
| PVPO    | Filter + GT | Mixed quality |
| XRPO    | UCB | Uncertainty + exploration |

Implementation Notes:
--------------------
- Requires scipy for Student's t-distribution
- ICL corpus should be curated high-quality examples
- Monitor priority distribution to tune λ
- Can combine with other filters (e.g., XRPO + DAPO filtering)
- Training step counter must be synchronized with trainer

Edge Cases:
----------
- n_q = 1: No std dev → use high default priority
- Zero variance: Δ̂_q = 0, rely on φ_q only
- No ICL corpus: Skip seeding, just use priority sampling
- All prompts zero reward: ICL seed all selected prompts
"""
