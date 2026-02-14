"""
Uniform Batch Sampler

Samples rollouts uniformly at random from the buffer.

Purpose:
-------
Simplest sampling strategy - every rollout has equal probability
of being selected for training. This is the default for most GRPO variants.

Role in Training Pipeline:
-------------------------
**Config**: `sampler: uniform`
**Step**: Called every training iteration to select training batch
**Algorithm**: Random sampling without replacement within batch

Key Responsibilities:
--------------------
1. Randomly select B prompts from buffer
2. For each prompt, get all G completions
3. Pad sequences to uniform length T
4. Stack into (B, G, T) tensors
5. Return TrainingBatch

Sampling Algorithm:
------------------
```
1. Get all unique prompt_ids from buffer
2. Randomly sample B prompt_ids
3. For each prompt_id, retrieve G rollouts
4. Validate all rollouts have rewards
5. Find max_length across all rollouts
6. Pad token_ids, log_probs, mask to max_length
7. Stack into batch tensors
8. Return TrainingBatch
```


Example Usage:
-------------
```python
# Create sampler
sampler = UniformBatchSampler(
    group_size=8,
    pad_token_id=0
)

# Sample batch
batch = sampler.sample(buffer, batch_size=4)

# batch.token_ids.shape = (4, 8, max_length)
# batch.rewards.shape = (4, 8)
```

Implementation Notes:
--------------------
- Use `random.sample()` for sampling without replacement
- Ensure deterministic sampling by setting random seed
- Handle edge case: fewer rollouts than group_size
- Validate all rewards are non-None
- Efficient padding: use torch operations
- Optional: cache padded tensors if sampling same rollouts

Edge Cases:
----------
- Buffer smaller than batch_size → raise ValueError
- Rollout group size mismatch → log warning, pad with dummy
- All sequences same length → no padding needed
- max_length exceeded → truncate from right
"""
