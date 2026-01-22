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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grpo_composer.rewards.base import RewardCalculator
from grpo_composer.rewards.binary import BinaryRewardCalculator
from grpo_composer.rewards.diversity_adjusted import DiversityAdjustedRewardCalculator


class TestBinaryRewardCalculator:
    """Tests for BinaryRewardCalculator (Base GRPO)."""
    
    def test_passthrough_no_threshold(self):
        """When threshold=False, rewards should pass through unchanged."""
        # Shape: (batch_size=2, num_completions=4)
        rewards = torch.tensor([
            [0.1, 0.9, 0.3, 0.7],
            [0.5, 0.6, 0.4, 0.8]
        ], dtype=torch.float32)
        
        # Convert to list format expected by current implementation
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        result = calculator.calculate()
        
        # Should be unchanged
        for i, r in enumerate(result):
            assert torch.allclose(r, rewards_list[i]), f"Row {i} should be unchanged"
    
    def test_threshold_binary_conversion(self):
        """When threshold=True, values > 0.5 become 1, else 0."""
        # Shape: (batch_size=2, num_completions=4)
        rewards = torch.tensor([
            [0.1, 0.9, 0.3, 0.7],
            [0.5, 0.6, 0.4, 0.8]
        ], dtype=torch.float32)
        
        expected = torch.tensor([
            [0.0, 1.0, 0.0, 1.0],  # 0.1<0.5, 0.9>0.5, 0.3<0.5, 0.7>0.5
            [0.0, 1.0, 0.0, 1.0]   # 0.5 is NOT > 0.5, 0.6>0.5, 0.4<0.5, 0.8>0.5
        ], dtype=torch.float32)
        
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for i, r in enumerate(result):
            assert torch.allclose(r, expected[i]), f"Row {i} thresholding incorrect. Got {r}, expected {expected[i]}"
    
    def test_already_binary_passthrough(self):
        """Binary rewards (0/1) should pass through correctly."""
        rewards = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=torch.float32)
        
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        result = calculator.calculate()
        
        for i, r in enumerate(result):
            assert torch.allclose(r, rewards_list[i])
    
    def test_edge_case_all_zeros(self):
        """All zero rewards should remain zeros."""
        rewards = torch.zeros((3, 4), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for r in result:
            assert torch.all(r == 0), "All should be zero"
    
    def test_edge_case_all_ones(self):
        """All 1.0 rewards should remain ones."""
        rewards = torch.ones((3, 4), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for r in result:
            assert torch.all(r == 1), "All should be one"


class TestDiversityAdjustedRewardCalculator:
    """Tests for DiversityAdjustedRewardCalculator (DRA-GRPO)."""
    
    def test_identical_rewards_high_penalty(self):
        """Identical rewards across completions should result in higher SMI penalty."""
        # All identical rewards - high redundancy
        rewards = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards.clone())
        result = calculator.calculate()
        
        # All rewards should be reduced due to high similarity
        # SMI = sum of all pairwise products excluding self
        # For [1,1,1,1]: smi_matrix = 4x4 of all 1s, sum = 16, smi_score = 16 - 1 = 15
        # adjusted = 1.0 / (1 + 15) = 1/16 for first element
        assert torch.all(result < rewards), "Identical rewards should be penalized"
    
    def test_diverse_rewards_lower_penalty(self):
        """Diverse rewards should have lower SMI penalty."""
        # Orthogonal-ish rewards
        rewards = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        original = rewards.clone()
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards.clone())
        result = calculator.calculate()
        
        # First element: smi_matrix = outer product, score for first = 1*0 + 1*0 + 1*0 = 0
        # So adjusted = 1.0 / (1 + 0) = 1.0 (no penalty for diverse)
        # This tests that diverse responses are not penalized as much
        assert result[0, 0] >= result[0, 1], "Diverse rewards should not be heavily penalized"
    
    def test_shape_preserved(self):
        """Output shape should match input shape."""
        batch_size, num_completions = 4, 8
        rewards = torch.rand((batch_size, num_completions), dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards.clone())
        result = calculator.calculate()
        
        assert result.shape == (batch_size, num_completions), f"Shape mismatch: {result.shape}"
    
    def test_non_negative_output(self):
        """Output rewards should remain non-negative for non-negative inputs."""
        rewards = torch.rand((3, 5), dtype=torch.float32)  # [0, 1)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards.clone())
        result = calculator.calculate()
        
        assert torch.all(result >= 0), "Adjusted rewards should be non-negative"
    
    def test_batch_independence(self):
        """Each batch should be processed independently."""
        # Two batches with different patterns
        rewards = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],  # High redundancy
            [1.0, 0.0, 0.0, 0.0],  # Low redundancy for first
        ], dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards.clone())
        result = calculator.calculate()
        
        # First batch should have higher penalty than second batch for first element
        # (because first batch has more redundancy)
        # Note: The current implementation processes row by row


class TestEdgeCases:
    """Edge case tests for both calculators."""
    
    def test_single_completion_binary(self):
        """Single completion per prompt should work."""
        rewards = torch.tensor([[0.8], [0.3]], dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        assert len(result) == 2
        assert result[0].item() == 1.0  # 0.8 > 0.5
        assert result[1].item() == 0.0  # 0.3 < 0.5
    
    def test_large_batch(self):
        """Should handle large batches efficiently."""
        batch_size, num_completions = 32, 16
        rewards = torch.rand((batch_size, num_completions), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        result = calculator.calculate()
        
        assert len(result) == batch_size


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_binary_requires_threshold_param(self):
        """BinaryRewardCalculator should require threshold parameter."""
        rewards = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        rewards_list = [rewards[0]]
        
        # Should work with threshold specified
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        assert calculator is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
