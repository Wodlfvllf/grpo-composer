
# We are starting to build a reward system for GRPO
# We Have decoupled training loop into multiple components. Reward calculation is one of them.
# Here we define the base calculation for GRPO rewards. Which then would be used later in advnatage calculation which is another component.

from abc import ABC, abstractmethod

class RewardCalculator(ABC):
    def __init__(self, rewards: list, **kwargs):
        self.rewards = rewards

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def calculate(self):
        pass
