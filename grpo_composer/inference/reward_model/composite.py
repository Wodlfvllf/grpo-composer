"""
Composite Reward Evaluator

Combines multiple reward signals with weighted sum.

What it does:
------------
- Runs multiple evaluators in parallel
- Combines rewards: r_final = Σ w_i * r_i
- Allows multi-objective optimization

When to use:
-----------
- Tasks with multiple quality dimensions
- Balance correctness + style + safety
- Custom weighted combinations

Example:
-------
```python
evaluator = CompositeRewardEvaluator(
    evaluators=[
        RuleBasedEvaluator(),    # Correctness
        LengthPenalty(),         # Brevity
        SafetyChecker()          # Harmlessness
    ],
    weights=[1.0, -0.1, 0.5]
)

rewards = evaluator.compute_rewards(prompts, completions)
# Returns weighted combination of all three signals
```

Common combinations:
- Math: correctness + length penalty
- Code: test pass + style + efficiency
- Chat: helpfulness + safety + engagement
"""
