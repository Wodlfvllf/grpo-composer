"""
GRPO Composer - Main Package

Modular GRPO framework for LLM reinforcement learning.
"""

from .core.rewards import (
    RewardCalculator,
    BinaryRewardCalculator,
    DiversityAdjustedRewardCalculator,
    FrequencyAwareRewardCalculator,
    LengthDependentRewardCalculator,
    PosteriorCompositeRewardCalculator,
    MultiRewardProcessor,
    RewardConfig,
)
from .core.rewards.rank_enhanced import RankEnhancedRewardCalculator
from .core.rewards.unlikeliness import UnlikelinessRewardCalculator
from .core.rewards.rts_based import RTSRewardCalculator

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
    "RankEnhancedRewardCalculator",
    "UnlikelinessRewardCalculator",
    "RTSRewardCalculator",
]

__version__ = "0.1.0"
