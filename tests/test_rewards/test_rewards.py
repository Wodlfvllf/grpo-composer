"""
Unit tests for Reward Calculators in grpo_composer.

Input format: rewards tensor of shape (batch_size, num_completions/group_size)
- batch_size: number of prompts in batch
- num_completions: number of responses per prompt (G in GRPO notation)
"""
import pytest
import torch
import torch.nn.functional as F

from grpo_composer import RewardCalculator
from grpo_composer import BinaryRewardCalculator
from grpo_composer import DiversityAdjustedRewardCalculator


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
    
    def test_threshold_binary_conversion(self):
        """When threshold=True, values > 0.5 become 1, else 0."""
        rewards = torch.tensor([
            [0.1, 0.9, 0.3, 0.7],
            [0.5, 0.6, 0.4, 0.8]
        ], dtype=torch.float32)
        
        expected = torch.tensor([
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for i, r in enumerate(result):
            assert torch.allclose(r, expected[i]), \
                f"Row {i} thresholding incorrect. Got {r}, expected {expected[i]}"
    
    def test_threshold_exactly_half(self):
        """Edge case: 0.5 should become 0 (not strictly greater than 0.5)."""
        rewards = torch.tensor([[0.5, 0.50001, 0.49999]], dtype=torch.float32)
        expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        
        calculator = BinaryRewardCalculator(rewards=[rewards[0]], threshold=True)
        result = calculator.calculate()
        
        assert torch.allclose(result[0], expected[0]), \
            f"0.5 should be 0, 0.50001 should be 1. Got {result[0]}"
    
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
            assert torch.allclose(r, rewards_list[i]), \
                f"Binary rewards changed unexpectedly at row {i}"
    
    def test_edge_case_all_zeros(self):
        """All zero rewards should remain zeros."""
        rewards = torch.zeros((3, 4), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for r in result:
            assert torch.all(r == 0), "All zeros should remain zeros"
    
    def test_edge_case_all_ones(self):
        """All 1.0 rewards should remain ones with threshold."""
        rewards = torch.ones((3, 4), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        for r in result:
            assert torch.all(r == 1), "All ones should remain ones"
    
    def test_negative_rewards(self):
        """Negative rewards should become 0 with threshold."""
        rewards = torch.tensor([[-0.5, 0.8, -1.0, 1.0]], dtype=torch.float32)
        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
        
        calculator = BinaryRewardCalculator(rewards=[rewards[0]], threshold=True)
        result = calculator.calculate()
        
        assert torch.allclose(result[0], expected[0]), \
            f"Negative values should become 0. Got {result[0]}"


class TestDiversityAdjustedRewardCalculator:
    """Tests for DiversityAdjustedRewardCalculator (DRA-GRPO)."""
    
    def test_identical_embeddings_high_penalty(self):
        """Identical embeddings should result in maximum penalty."""
        rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # All identical embeddings
        embeddings = torch.ones((1, 4, 128), dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # All embeddings identical → sim_matrix all 1s
        # SMI for each = sum(row) - diagonal = 4 - 1 = 3
        # adjusted = 1.0 / (1 + 3) = 0.25
        expected = torch.full_like(rewards, 0.25)
        
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Identical embeddings should yield {expected}, got {result}"
    
    def test_orthogonal_embeddings_no_penalty(self):
        """Orthogonal embeddings should have minimal penalty."""
        rewards = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # Orthogonal embeddings (zero cosine similarity)
        embeddings = torch.zeros((1, 3, 3), dtype=torch.float32)
        embeddings[0, 0, 0] = 1.0  # [1, 0, 0]
        embeddings[0, 1, 1] = 1.0  # [0, 1, 0]
        embeddings[0, 2, 2] = 1.0  # [0, 0, 1]
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # Orthogonal → sim_matrix is identity matrix
        # SMI for each = 0 (no similarity to others)
        # adjusted = 1.0 / (1 + 0) = 1.0
        assert torch.allclose(result, rewards, atol=1e-4), \
            f"Orthogonal embeddings should have no penalty. Expected {rewards}, got {result}"
    
    def test_partial_similarity(self):
        """Test with known partial similarity between embeddings."""
        rewards = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # Embeddings with controlled similarity
        # e1 = [1, 0], e2 = [0.6, 0.8] (60° angle, cos=0.6), e3 = [0, 1] (90° to e1)
        embeddings = torch.tensor([
            [
                [1.0, 0.0],
                [0.6, 0.8],
                [0.0, 1.0],
            ]
        ], dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # Manually compute expected
        # Normalize embeddings
        e_norm = F.normalize(embeddings[0], p=2, dim=-1)
        sim_matrix = e_norm @ e_norm.T
        
        # SMI: sum of row - diagonal
        smi = sim_matrix.sum(dim=-1) - torch.diagonal(sim_matrix)
        expected = rewards[0] / (1.0 + smi + 1e-6)
        
        assert torch.allclose(result[0], expected, atol=1e-3), \
            f"Partial similarity adjustment incorrect. Expected {expected}, got {result[0]}"
    
    def test_shape_preserved(self):
        """Output shape should match input shape."""
        batch_size, num_completions, hidden_size = 4, 8, 128
        rewards = torch.rand((batch_size, num_completions), dtype=torch.float32)
        embeddings = torch.randn((batch_size, num_completions, hidden_size))
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        assert result.shape == (batch_size, num_completions), \
            f"Shape mismatch: expected {rewards.shape}, got {result.shape}"
    
    def test_non_negative_output(self):
        """Output rewards should remain non-negative for non-negative inputs."""
        rewards = torch.rand((3, 5), dtype=torch.float32)
        embeddings = torch.randn((3, 5, 128))
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        assert torch.all(result >= 0), \
            f"Adjusted rewards should be non-negative, got min={result.min()}"
    
    def test_zero_rewards_remain_zero(self):
        """Zero rewards should remain zero regardless of embeddings."""
        rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
        embeddings = torch.randn((1, 4, 128))
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        assert result[0, 0] == 0.0 and result[0, 2] == 0.0, \
            f"Zero rewards should stay zero. Got {result}"
    
    def test_batch_independence(self):
        """Each batch should be processed independently."""
        rewards = torch.tensor([
            [1.0, 1.0, 1.0],  # Batch 1
            [1.0, 1.0, 1.0],  # Batch 2
        ], dtype=torch.float32)
        
        embeddings = torch.zeros((2, 3, 3))
        # Batch 1: identical embeddings (high penalty)
        embeddings[0, :, :] = 1.0
        # Batch 2: orthogonal embeddings (no penalty)
        embeddings[1, 0, 0] = 1.0
        embeddings[1, 1, 1] = 1.0
        embeddings[1, 2, 2] = 1.0
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # Batch 1 should have lower rewards (high penalty)
        # Batch 2 should have higher rewards (no penalty)
        assert result[0, 0] < result[1, 0], \
            "Batch with identical embeddings should have lower adjusted rewards"
    
    def test_higher_similarity_lower_reward(self):
        """Completions with higher similarity should get lower adjusted rewards."""
        rewards = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        
        # Two identical embeddings (maximum similarity)
        embeddings = torch.ones((1, 2, 128), dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # SMI = 1 (similarity between the two)
        # adjusted = 1.0 / (1 + 1) = 0.5
        expected = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        
        assert torch.allclose(result, expected, atol=1e-4), \
            f"High similarity should reduce rewards. Expected {expected}, got {result}"
    
    def test_smi_formula_correctness(self):
        """Verify SMI formula: SMI(o_i) = sum_{j != i} similarity(o_i, o_j)."""
        rewards = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        
        # Create embeddings with known similarities
        embeddings = torch.tensor([
            [
                [1.0, 0.0],  # e1
                [0.8, 0.6],  # e2
                [0.0, 1.0],  # e3
            ]
        ], dtype=torch.float32)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        
        # Manually compute expected SMI
        e_norm = F.normalize(embeddings[0], p=2, dim=-1)
        sim_matrix = e_norm @ e_norm.T
        
        # sim_matrix:
        # [[1.0, 0.8, 0.0],
        #  [0.8, 1.0, 0.6],
        #  [0.0, 0.6, 1.0]]
        
        # SMI[0] = 0.8 + 0.0 = 0.8
        # SMI[1] = 0.8 + 0.6 = 1.4
        # SMI[2] = 0.0 + 0.6 = 0.6
        
        smi_expected = sim_matrix.sum(dim=-1) - torch.diagonal(sim_matrix)
        
        # Get actual SMI from calculator
        smi_actual = calculator._calculate_smi()
        
        assert torch.allclose(smi_actual[0], smi_expected, atol=1e-4), \
            f"SMI calculation incorrect. Expected {smi_expected}, got {smi_actual[0]}"


class TestEdgeCases:
    """Edge case tests for both calculators."""
    
    def test_single_completion_binary(self):
        """Single completion per prompt should work."""
        rewards = torch.tensor([[0.8], [0.3]], dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        result = calculator.calculate()
        
        assert len(result) == 2
        assert result[0].item() == 1.0, "0.8 > 0.5 should be 1.0"
        assert result[1].item() == 0.0, "0.3 < 0.5 should be 0.0"
    
    def test_single_completion_diversity(self):
        """Single completion should have no diversity penalty (SMI = 0)."""
        rewards = torch.tensor([[1.0]], dtype=torch.float32)
        embeddings = torch.randn((1, 1, 128))
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # Single completion → no other completions to compare
        # SMI = 0 → adjusted = 1.0 / (1 + 0) = 1.0
        assert torch.allclose(result, rewards, atol=1e-4), \
            f"Single completion should have no penalty. Expected {rewards}, got {result}"
    
    def test_large_batch(self):
        """Should handle large batches efficiently."""
        batch_size, num_completions = 32, 16
        rewards = torch.rand((batch_size, num_completions), dtype=torch.float32)
        rewards_list = [rewards[i] for i in range(rewards.shape[0])]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        result = calculator.calculate()
        
        assert len(result) == batch_size, \
            f"Expected {batch_size} batches, got {len(result)}"


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_binary_requires_threshold_param(self):
        """BinaryRewardCalculator should handle threshold parameter."""
        rewards = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        rewards_list = [rewards[0]]
        
        calculator = BinaryRewardCalculator(rewards=rewards_list, threshold=False)
        assert calculator is not None
        
        calculator_thresh = BinaryRewardCalculator(rewards=rewards_list, threshold=True)
        assert calculator_thresh is not None
    
    def test_diversity_shape_validation(self):
        """DiversityAdjustedRewardCalculator should validate input shapes."""
        # Mismatched batch sizes
        rewards = torch.rand((4, 8))
        embeddings = torch.randn((3, 8, 128))  # Wrong batch size
        
        with pytest.raises(AssertionError, match="Batch size mismatch"):
            calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
    
    def test_diversity_group_size_validation(self):
        """DiversityAdjustedRewardCalculator should validate group sizes."""
        # Mismatched group sizes
        rewards = torch.rand((4, 8))
        embeddings = torch.randn((4, 6, 128))  # Wrong group size
        
        with pytest.raises(AssertionError, match="Group size mismatch"):
            calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
    
    def test_diversity_wrong_dimensions(self):
        """DiversityAdjustedRewardCalculator should reject wrong tensor dimensions."""
        # Wrong reward dimensions
        rewards = torch.rand((4, 8, 2))  # 3D instead of 2D
        embeddings = torch.randn((4, 8, 128))
        
        with pytest.raises(AssertionError, match="Rewards must be 2D"):
            calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)


class TestDRAGRPOFormula:
    """Tests specifically for DRA-GRPO formula: adjusted = reward / (1 + SMI)."""
    
    def test_formula_with_zero_smi(self):
        """When SMI = 0 (orthogonal), adjusted should equal original."""
        rewards = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        
        # Orthogonal embeddings → SMI = 0
        embeddings = torch.eye(3).unsqueeze(0)  # Identity matrix
        # Shape: (1, 3, 3)
        
        calculator = DiversityAdjustedRewardCalculator(rewards=rewards, embedding=embeddings)
        result = calculator.calculate()
        
        # SMI = 0 → adjusted = reward / (1 + 0) = reward
        assert torch.allclose(result, rewards, atol=1e-4), \
            f"Zero SMI should preserve rewards. Expected {rewards}, got {result}"
    
    def test_formula_with_known_smi(self):
        """Test formula with manually computed SMI values."""
        rewards = torch.tensor([[2.0]], dtype=torch.float32)
        
        # Two identical embeddings → similarity = 1.0
        embeddings = torch.ones((1, 2, 128))
        
        calculator = DiversityAdjustedRewardCalculator(
            rewards=torch.tensor([[2.0, 2.0]]), 
            embedding=embeddings
        )
        result = calculator.calculate()
        
        # SMI[0] = similarity(e0, e1) = 1.0
        # adjusted[0] = 2.0 / (1 + 1.0) = 1.0
        expected = torch.tensor([[1.0, 1.0]])
        
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Formula mismatch. Expected {expected}, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
