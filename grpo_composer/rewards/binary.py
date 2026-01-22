"""
Binary Reward Calculator
This class Implements Binary Reward Calculator for Base GRPO class to 
calculate binary rewards which would then be used in Advantage Calculation.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import RewardCalculator

class BinaryRewardCalculator(RewardCalculator):
    def __init__(self, rewards: list, threshold: bool, **kwargs):
        super().__init__(rewards, **kwargs) 
        self.rewards = rewards
        self.threshold = threshold

    def calculate(self):
        if self.threshold:
            return [torch.where(reward > 0.5, torch.ones_like(reward), torch.zeros_like(reward)) for reward in self.rewards]
        else:
            return self.rewards


