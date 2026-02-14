"""
Reward Model Interface

Abstract base class for reward computation.

Defines:
-------
- `RewardEvaluator(ABC)`: Interface for computing rewards

Implemented by:
--------------
- `inference/reward_model/rule_based.py` - Verifiers
- `inference/reward_model/learned.py` - Neural RM
- `inference/reward_model/composite.py` - Multi-objective

Key method:
----------
```python
def compute_rewards(
    self,
    prompts: List[str],
    completions: List[str]
) -> torch.Tensor:  # (B*G,)
    '''Return scalar reward for each completion.'''
    pass
```
"""
