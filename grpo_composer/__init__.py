"""
GRPO Composer - Main Package

Modular GRPO framework for LLM reinforcement learning.
"""

from .rewards import (
    RewardCalculator,
    BinaryRewardCalculator,
    DiversityAdjustedRewardCalculator,
    FrequencyAwareRewardCalculator,
    LengthDependentRewardCalculator,
    PosteriorCompositeRewardCalculator,
    MultiRewardProcessor,
    RewardConfig,
)

__all__ = [
    # Reward Calculators
    "RewardCalculator",
    "BinaryRewardCalculator",
    "DiversityAdjustedRewardCalculator",
    "FrequencyAwareRewardCalculator",
    "LengthDependentRewardCalculator",
    "PosteriorCompositeRewardCalculator",
    "MultiRewardProcessor",
    "RewardConfig",
]

__version__ = "0.1.0"