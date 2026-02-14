"""
HuggingFace Trainer Integration

Adapter to use GRPO with HuggingFace Trainer API.

What it does:
------------
- Wraps our GRPOTrainer to work with HF Trainer
- Allows using HF ecosystemfeatures (callbacks, logging)
- Compatible with existing HF training scripts

When to use:
-----------
- Existing HF Trainer pipelines
- Want HF integrations (Weights & Biases, TensorBoard)
- Familiar with HF Trainer API

Example:
-------
```python
from grpo_composer.integrations import HFGRPOTrainer

trainer = HFGRPOTrainer(
    model=policy_model,
    args=training_args,
    grpo_config=grpo_config,
    train_dataset=dataset
)

trainer.train()  # Uses HF Trainer loop with GRPO logic
```

Bridges our GRPO implementation with HF ecosystem.
"""
