"""
Unit tests for Reward Calculators in grpo_composer.

Input format: rewards tensor of shape (batch_size, num_completions/group_size)
- batch_size: number of prompts in batch
- num_completions: number of responses per prompt (G in GRPO notation)
"""
import pytest
import torch
import sys
import os

# Add parent directory to path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grpo_composer.rewards import RewardCalculator
from grpo_composer.rewards import BinaryRewardCalculator
from grpo_composer.rewards import DiversityAdjustedRewardCalculator


class TestBinaryRewardCalculator:
    """Tests for BinaryRewardCalculator (Base GRPO)."""
    
    def test_passthrough_no_threshold(self):
        """When threshold=False, rewards should pass through unchanged."""
        rewards = torch.tensor([
            [0.1, 0.9, 0.3, 0.7],
            [0.5, 0.6, 0.4, 0.8]
        ], dtype=torch.float32)
        
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        result = calculator.calculate()
        
        for i, r in enumerate(result):
            assert torch.allclose(r, rewards_list[i]), \
                f"Row {i} should be unchanged. Got {r}, expected {rewards_list[i]}"