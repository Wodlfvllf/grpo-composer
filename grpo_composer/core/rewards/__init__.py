"""
GRPO Composer - Reward Calculators

This module provides various reward calculation strategies for GRPO training.
"""

from .base import RewardCalculator
from .binary import BinaryRewardCalculator
from .diversity_adjusted import DiversityAdjustedRewardCalculator
from .frequency_aware import FrequencyAwareRewardCalculator
from .length_dependent import LengthDependentRewardCalculator
from .posterior_composite import PosteriorCompositeRewardCalculator
from .multi_reward import MultiRewardProcessor, RewardConfig

__all__ = [
    # Base
    "RewardCalculator",
    # Paper implementations
    "BinaryRewardCalculator",           # Base GRPO
    "DiversityAdjustedRewardCalculator", # DRA-GRPO
    "FrequencyAwareRewardCalculator",    # GAPO
    "LengthDependentRewardCalculator",   # GRPO-LEAD
    "PosteriorCompositeRewardCalculator", # P-GRPO
    "MultiRewardProcessor",              # GDPO
    "RewardConfig",
]