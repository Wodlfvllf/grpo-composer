"""
Rule-Based Reward Evaluator

Uses deterministic verifiers (code execution, answer checking, parsers).

What it does:
------------
- Extracts final answer from completion
- Compares with ground truth or executes code
- Returns binary reward (0 or 1) or scaled score

When to use:
-----------
- Math problems (answer extraction + verification)
- Code generation (unit test execution)
- Tasks with objective ground truth

Example:
-------
```python
evaluator = RuleBasedEvaluator(
    task_type="math",
    ground_truth_file="gsm8k_answers.json"
)

# Completions: ["So the answer is 42", "The answer is 43"]
rewards = evaluator.compute_rewards(prompts, completions)
# Returns: [1.0, 0.0] (first correct, second wrong)
```

Supports:
- Math: Parse answer, compare numerically
- Code: Execute and check test cases
- QA: Exact match or fuzzy match
"""
