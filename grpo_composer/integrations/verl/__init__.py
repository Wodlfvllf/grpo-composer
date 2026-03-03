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
from .trainer import ComposerRayPPOTrainer, patch_verl_main_ppo, unpatch_verl_main_ppo

__all__ = [
    "advantages",
    "losses",
    "ComposerRayPPOTrainer",
    "patch_verl_main_ppo",
    "unpatch_verl_main_ppo",
]
