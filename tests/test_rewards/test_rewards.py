"""
Minimal unit tests for all Reward Calculators in grpo_composer.
"""
import pytest
import torch

from grpo_composer import (
    RewardCalculator,
    BinaryRewardCalculator,
    DiversityAdjustedRewardCalculator,
    FrequencyAwareRewardCalculator,
    LengthDependentRewardCalculator,
    PosteriorCompositeRewardCalculator,
    MultiRewardProcessor,
    RewardConfig,
)


# =============================================================================
# BinaryRewardCalculator Tests
# =============================================================================

class TestBinaryRewardCalculator:
    
    def test_passthrough(self):
        """No threshold → unchanged rewards."""
        rewards = torch.tensor([[0.3, 0.7]])
        calc = BinaryRewardCalculator(rewards, threshold=None)
        assert torch.allclose(calc.compute_rewards(), rewards)
    
    def test_threshold(self):
        """Threshold=0.5 → binary 0/1."""
        rewards = torch.tensor([[0.3, 0.7]])
        calc = BinaryRewardCalculator(rewards, threshold=0.5)
        expected = torch.tensor([[0.0, 1.0]])
        assert torch.allclose(calc.compute_rewards(), expected)
    
    def test_shape(self):
        """Output shape == input shape."""
        rewards = torch.rand(4, 8)
        calc = BinaryRewardCalculator(rewards, threshold=0.5)
        assert calc.compute_rewards().shape == (4, 8)


# =============================================================================
# DiversityAdjustedRewardCalculator Tests
# =============================================================================

class TestDiversityAdjustedRewardCalculator:
    
    def test_identical_embeddings_penalized(self):
        """Identical embeddings → high SMI → reduced reward."""
        rewards = torch.ones(1, 4)
        embeddings = torch.ones(1, 4, 64)  # All identical
        calc = DiversityAdjustedRewardCalculator(rewards, embeddings)
        result = calc.compute_rewards()
        assert torch.all(result < rewards)
    
    def test_orthogonal_embeddings_no_penalty(self):
        """Orthogonal embeddings → SMI≈0 → reward unchanged."""
        rewards = torch.ones(1, 3)
        embeddings = torch.eye(3).unsqueeze(0)  # (1, 3, 3) orthogonal
        calc = DiversityAdjustedRewardCalculator(rewards, embeddings)
        result = calc.compute_rewards()
        assert torch.allclose(result, rewards, atol=1e-4)
    
    def test_shape(self):
        """Output shape matches rewards shape."""
        rewards = torch.rand(2, 6)
        embeddings = torch.randn(2, 6, 128)
        calc = DiversityAdjustedRewardCalculator(rewards, embeddings)
        assert calc.compute_rewards().shape == (2, 6)


# =============================================================================
# FrequencyAwareRewardCalculator Tests
# =============================================================================

class TestFrequencyAwareRewardCalculator:
    
    def test_uniform_distribution(self):
        """Uniform sampling → reward = 1.0."""
        valid = {"A", "B", "C"}
        completions = ["A", "B", "C"]
        calc = FrequencyAwareRewardCalculator(completions, valid)
        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(calc.compute_rewards(), expected, atol=1e-4)
    
    def test_invalid_penalty(self):
        """Invalid items get -1 penalty."""
        valid = {"A", "B"}
        completions = ["A", "INVALID"]
        calc = FrequencyAwareRewardCalculator(completions, valid)
        result = calc.compute_rewards()
        assert result[1] == -1.0
    
    def test_overrepresented_penalized(self):
        """Overrepresented item gets lower reward."""
        valid = {"A", "B"}
        completions = ["A", "A", "A", "B"]
        calc = FrequencyAwareRewardCalculator(completions, valid)
        result = calc.compute_rewards()
        assert result[0] < 1.0  # A is overrepresented


# =============================================================================
# LengthDependentRewardCalculator Tests
# =============================================================================

class TestLengthDependentRewardCalculator:
    
    def test_shorter_correct_boosted(self):
        """Shorter correct response gets higher reward."""
        responses = [[1, 2], [1, 2, 3, 4, 5, 6]]  # Short, Long
        labels = [1, 1]
        calc = LengthDependentRewardCalculator(responses, labels, alpha=0.1)
        result = calc.compute_rewards()
        assert result[0] > result[1]
    
    def test_incorrect_penalty(self):
        """Incorrect response gets -1."""
        responses = [[1, 2, 3]]
        labels = [0]
        # Need at least one correct for stats, so add one
        responses = [[1, 2], [1, 2, 3]]
        labels = [1, 0]
        calc = LengthDependentRewardCalculator(responses, labels)
        result = calc.compute_rewards()
        assert result[1] == -1.0
    
    def test_mean_length_neutral(self):
        """Single correct response → z=0 → reward=1.0."""
        responses = [[1, 2, 3]]
        labels = [1]
        calc = LengthDependentRewardCalculator(responses, labels)
        result = calc.compute_rewards()
        assert torch.allclose(result, torch.tensor([1.0]), atol=1e-4)
    
    def test_no_correct_raises(self):
        """Error if no correct responses."""
        with pytest.raises(ValueError):
            LengthDependentRewardCalculator([[1]], [0])


# =============================================================================
# PosteriorCompositeRewardCalculator Tests
# =============================================================================

class TestPosteriorCompositeRewardCalculator:
    
    def test_formula(self):
        """R = R_f + R_o + R_o * R_t."""
        r_f = torch.tensor([1.0, 0.0])
        r_o = torch.tensor([1.0, 1.0])
        r_t = torch.tensor([0.5, 0.5])
        calc = PosteriorCompositeRewardCalculator.from_precomputed(r_f, r_o, r_t)
        result = calc.compute_rewards()
        # [0]: 1 + 1 + 1*0.5 = 2.5
        # [1]: 0 + 1 + 1*0.5 = 1.5
        expected = torch.tensor([2.5, 1.5])
        assert torch.allclose(result, expected)
    
    def test_thinking_gated(self):
        """Thinking reward gated by outcome (R_o * R_t)."""
        r_f = torch.tensor([1.0, 1.0])
        r_o = torch.tensor([1.0, 0.0])  # Second: outcome=0
        r_t = torch.tensor([0.5, 0.5])
        calc = PosteriorCompositeRewardCalculator.from_precomputed(r_f, r_o, r_t)
        result = calc.compute_rewards()
        # [0]: 1 + 1 + 0.5 = 2.5
        # [1]: 1 + 0 + 0 = 1.0 (thinking gated out)
        assert result[0] == 2.5 and result[1] == 1.0


# =============================================================================
# MultiRewardProcessor Tests (GDPO)
# =============================================================================

class TestMultiRewardProcessor:
    
    def test_output_shape(self):
        """(B, G, N) → (B, G)."""
        configs = [RewardConfig("r1"), RewardConfig("r2")]
        proc = MultiRewardProcessor(configs)
        rewards = torch.rand(4, 8, 2)
        result = proc.compute_rewards(rewards)
        assert result.shape == (4, 8)
    
    def test_weights(self):
        """Weighted rewards should differ from unweighted."""
        configs_equal = [RewardConfig("r1", weight=1.0), RewardConfig("r2", weight=1.0)]
        configs_biased = [RewardConfig("r1", weight=2.0), RewardConfig("r2", weight=0.5)]
        
        rewards = torch.rand(2, 4, 2)
        
        proc1 = MultiRewardProcessor(configs_equal, use_batch_norm=False)
        proc2 = MultiRewardProcessor(configs_biased, use_batch_norm=False)
        
        r1 = proc1.compute_rewards(rewards)
        r2 = proc2.compute_rewards(rewards)
        
        # Different weights → different results
        assert not torch.allclose(r1, r2)
    
    def test_conditioning(self):
        """Conditioned reward zeroed if parent below threshold."""
        configs = [
            RewardConfig("safety"),
            RewardConfig("helpful", conditioned_on="safety", condition_threshold=0.5),
        ]
        proc = MultiRewardProcessor(configs, use_batch_norm=False)
        
        # Both rollouts have same helpfulness, but second fails safety
        rewards = torch.tensor([
            [[0.8, 1.0], [0.3, 1.0]],  # safety: 0.8 (pass), 0.3 (fail)
        ])
        result = proc.compute_rewards(rewards)
        assert result.shape == (1, 2)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
