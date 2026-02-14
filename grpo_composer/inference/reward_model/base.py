"""
Base Reward Evaluator Interface

Defines how to compute rewards for generated completions.

Purpose:
-------
Score completions to produce reward signal for RL optimization.

Interface:
---------
```python
class RewardEvaluator(ABC):
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str]
    ) -> torch.Tensor:  # (B*G,) rewards
        '''Return scalar reward for each completion.'''
        pass
```

Used in training:
```python
rewards = reward_evaluator.compute_rewards(prompts, completions)
advantages = rewards - rewards.mean()  # Center advantages
```
"""
