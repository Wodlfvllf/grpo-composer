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
from . import rewards_registery
from .trainer import ComposerRayPPOTrainer
from .entrypoint import ComposerTaskRunner, run

__all__ = [
    "advantages",
    "losses",
    "utils",
    "clip_registery",
    "aggregations_registery",
    "regularisation_registery",
    "rewards_registery",
    "ComposerRayPPOTrainer",
    "ComposerTaskRunner",
    "run",
]
