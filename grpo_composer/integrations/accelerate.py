"""
HuggingFace Accelerate Integration

Adapter to use GRPO with Accelerate for distributed training.

What it does:
------------
- Wraps training loop with Accelerate
- Handles multi-GPU, multi-node setup automatically
- Simplifies distributed code

When to use:
-----------
- Simple distributed training (without FSDP/DeepSpeed complexity)
- Quick multi-GPU setup
- Prototyping distributed training

Example:
-------
```python
from accelerate import Accelerator
from grpo_composer.integrations import AccelerateGRPOTrainer

accelerator = Accelerator()

trainer = AccelerateGRPOTrainer(
    model=policy_model,
    grpo_config=config,
    accelerator=accelerator
)

trainer.train()  # Automatically multi-GPU
```

Simplifies distributed GRPO without manual setup.
"""
