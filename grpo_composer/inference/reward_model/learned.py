"""
Learned Reward Model

Neural network trained to predict reward (preference model, Bradley-Terry).

What it does:
------------
- Forwards completion through reward model network
- Outputs scalar reward score
- Trained on human preferences or correctness labels

When to use:
-----------
- Subjective tasks (chat, creative writing, helpfulness)
- No objective ground truth available
- Alignment with human preferences needed

Example:
-------
```python
evaluator = LearnedRewardModel(
    model_path="OpenAssistant/reward-model-deberta-v3-large"
)

# Completions: ["Here's a helpful answer...", "Bad response"]
rewards = evaluator.compute_rewards(prompts, completions)
# Returns: [0.8, -0.3] (continuous scores)
```

Common reward models:
- OpenAssistant/reward-model-deberta
- weqweasdas/RM-Mistral-7B
- Custom trained on task-specific preferences
"""
