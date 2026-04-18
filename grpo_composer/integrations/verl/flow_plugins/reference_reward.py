"""PVPO/GAPO reference-reward generation as a :class:`FlowPlugin`.

Replaces the closure-based ``hooked_compute_advantage`` that previously
lived inside ``ComposerRayPPOTrainer.fit`` and was injected by swapping
``verl.trainer.ppo.ray_trainer.compute_advantage`` at the module level.

The trainer now calls :meth:`ReferenceRewardFlowPlugin.before_compute_advantage`
directly for each post-rollout DataProto, so we no longer mutate veRL's
module globals.
"""

from __future__ import annotations

import os

from ..ref_reward_runtime import ensure_reference_rewards, has_reference_rewards
from ..trainer import FlowPlugin


class ReferenceRewardFlowPlugin(FlowPlugin):
    """Generate reference-policy rollouts and attach their rewards to the batch.

    Activated for any flow that needs a reference baseline (PVPO, GAPO,
    or any custom flow whose name contains ``reference_rewards``).
    """

    def configure(self, trainer) -> None:
        composer_cfg = getattr(trainer, "composer_config_dict", {}) or {}
        self._composer_cfg = composer_cfg
        self._ref_reward_source = str(
            composer_cfg.get("reference_reward_source", "auto")
        ).strip().lower()

    def before_compute_advantage(self, trainer, data):
        if has_reference_rewards(data):
            return data
        debug = os.environ.get("GRPO_COMPOSER_DEBUG") == "1"
        ensure_reference_rewards(
            trainer,
            data,
            composer_cfg=self._composer_cfg,
            ref_reward_source=self._ref_reward_source,
            debug=debug,
        )
        return data
