"""
Trainer Interface

Abstract base class for GRPO training loop.

Defines:
-------
- `Trainer(ABC)`: Interface for orchestrating training

Key methods:
-----------
```python
def train_step(self, prompts: List[str]) -> Dict[str, float]:
    '''Execute one training iteration.'''
    pass

def train(self) -> None:
    '''Run full training loop.'''
    pass
```

Implemented by:
--------------
- `runtime/trainer/grpo_trainer.py` - Main GRPO trainer
- `runtime/trainer/distributed_trainer.py` - Multi-node

Training flow:
-------------
1. Sample prompts from dataset
2. Generate completions via engine
3. Compute rewards
4. Sample batch from buffer
5. Compute loss and optimize
6. Update reference model if needed
"""
