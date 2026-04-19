"""
grpo_composer ↔ veRL Integration

Importing this module registers all custom advantage estimators and
policy losses into veRL's registries.

Usage:
    # In your training script or veRL config:
    import grpo_composer.integrations.verl  # triggers registration

    # Then use in YAML config:
    #   algorithm.adv_estimator: "difficulty_aware_grpo"
    #   actor_rollout_ref.actor.loss_fn: "composer"
"""

from . import advantages  # registers @register_adv_est decorators
from . import losses      # registers @register_policy_loss decorators
from . import utils
from . import clip_registery
from . import aggregations_registery
from . import regularisation_registery
from . import rewards_registry
from .trainer import ComposerRayPPOTrainer
from .entrypoint import ComposerTaskRunner, run

# Backward-compatibility alias for the old typo'd module name. Will be
# removed in v0.2.0; new code should import `rewards_registry`.
rewards_registery = rewards_registry

__all__ = [
    "advantages",
    "losses",
    "utils",
    "clip_registery",
    "aggregations_registery",
    "regularisation_registery",
    "rewards_registry",
    "rewards_registery",  # deprecated alias
    "ComposerRayPPOTrainer",
    "ComposerTaskRunner",
    "run",
]
