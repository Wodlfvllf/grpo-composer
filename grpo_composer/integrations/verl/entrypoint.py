"""Composer launch entrypoint for veRL PPO training.

This module replaces the previous monkey-patch approach
(``patch_verl_main_ppo``) that mutated ``verl.trainer.main_ppo.RayPPOTrainer``
and ``verl.trainer.ppo.ray_trainer.compute_advantage`` at import time.

Two clean extensions:

* :class:`ComposerTaskRunner` — subclass of upstream :class:`TaskRunner` whose
  ``add_actor_rollout_worker`` registers
  :class:`ComposerActorRolloutRefWorker` (instead of the upstream
  ``ActorRolloutRefWorker``) under ``Role.ActorRollout`` *before* it is
  wrapped with ``ray.remote(...)``. ``run`` instantiates
  :class:`ComposerRayPPOTrainer` instead of the upstream trainer; advantage
  computation is overridden inside that subclass's ``fit`` method, so no
  module-global swap is required.
* :func:`run` — a one-liner that delegates to upstream ``run_ppo`` with
  ``task_runner_class=ray.remote(num_cpus=1)(ComposerTaskRunner)``.

Use from a launch script as::

    from grpo_composer.integrations.verl.entrypoint import run
    run(config)
"""

from __future__ import annotations

import socket
import os
from typing import Any, Optional

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import (
        TaskRunner,
        create_rl_dataset,
        create_rl_sampler,
        run_ppo,
    )
    from verl.trainer.ppo.ray_trainer import Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.config import validate_config
    from verl.utils.fs import copy_to_local
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    ray = None
    OmegaConf = None
    TaskRunner = object  # type: ignore[assignment]
    Role = None
    create_rl_dataset = None
    create_rl_sampler = None
    run_ppo = None
    load_reward_manager = None
    need_critic = None
    need_reference_policy = None
    hf_processor = None
    hf_tokenizer = None
    validate_config = None
    copy_to_local = None

from .composer_workers import ComposerActorRolloutRefWorker
from .trainer import ComposerRayPPOTrainer


class ComposerTaskRunner(TaskRunner):
    """TaskRunner that wires composer subclasses into veRL's launch flow.

    * Substitutes :class:`ComposerActorRolloutRefWorker` for the FSDP
      actor-rollout role so the worker uses :class:`ComposerDataParallelPPOActor`
      and surfaces hidden states from ``compute_log_prob``.
    * Instantiates :class:`ComposerRayPPOTrainer` (which carries the composer
      ``fit`` loop, flow plugins, and Tracking/CSV fallback) instead of the
      upstream :class:`RayPPOTrainer`.

    All other roles (critic, reward model, reference policy) stay on upstream
    workers; composer behavior is concentrated in the actor + trainer.
    """

    def add_actor_rollout_worker(self, config):  # type: ignore[override]
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup

            if config.actor_rollout_ref.rollout.mode == "async":
                # Async path uses upstream's AsyncActorRolloutRefWorker, which
                # currently is not subclassed by composer. Fall back to upstream
                # behavior so async runs still work; composer-required paths
                # (DRA-GRPO etc.) use sync rollout.
                return super().add_actor_rollout_worker(config)

            actor_rollout_cls = ComposerActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
            return actor_rollout_cls, ray_worker_group_cls

        # Megatron / other strategies: defer to upstream.
        return super().add_actor_rollout_worker(config)

    def run(self, config):  # type: ignore[override]
        """Mirror upstream ``TaskRunner.run`` but instantiate ``ComposerRayPPOTrainer``."""
        from pprint import pprint

        # Re-apply registry side-effects inside the Ray actor process. Imports
        # in the driver process do not propagate to spawned actors.
        import grpo_composer.integrations.verl  # noqa: F401

        print(f"ComposerTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ComposerRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def run(config: Any) -> None:
    """Launch composer PPO training. Replaces the legacy ``patch_verl_main_ppo`` flow."""
    if run_ppo is None:
        raise RuntimeError(
            "grpo_composer.integrations.verl.entrypoint.run requires `verl`. "
            f"Original import error: {_VERL_IMPORT_ERROR!r}"
        )
    task_runner_class = ray.remote(num_cpus=1)(ComposerTaskRunner)
    return run_ppo(config, task_runner_class=task_runner_class)


__all__ = ["ComposerTaskRunner", "run"]
