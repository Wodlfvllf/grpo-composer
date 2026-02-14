# grpo_composer/core/__init__.py
"""
Core mathematical components for GRPO loss computation.

This module contains all the algorithmic building blocks:
- advantages: Advantage calculation methods
- aggregation: Loss aggregation strategies  
- clipping: Ratio clipping mechanisms
- regularizers: KL, preference, MI regularization
- losses: Final loss composition
- rewards: Reward calculation functions
"""

from . import advantages
from . import aggregation
from . import clipping
from . import regularizers
from . import losses
from . import rewards

__all__ = [
    "advantages",
    "aggregation", 
    "clipping",
    "regularizers",
    "losses",
    "rewards",
]
