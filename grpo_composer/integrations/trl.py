"""
TRL (Transformer Reinforcement Learning) Integration

Adapter to use GRPO with TRL library (PPO, DPO alternative).

What it does:
------------
- Makes GRPO compatible with TRL's RL training framework
- Uses TRL's utilities (data collators, trackers)
- Alternative to TRL's PPOTrainer

When to use:
-----------
- Already using TRL ecosystem
- Want TRL's built-in features
- Comparing GRPO vs PPO/DPO

Example:
-------
```python
from grpo_composer.integrations import TRLGRPOTrainer

trainer = TRLGRPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    config=grpo_config
)

trainer.train()  # TRL-style API with GRPO algorithm
```

Bridges our GRPO with TRL's RL training ecosystem.
"""
