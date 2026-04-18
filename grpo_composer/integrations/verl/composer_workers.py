"""Composer subclass of veRL's :class:`ActorRolloutRefWorker`.

This module defines :class:`ComposerActorRolloutRefWorker`, which extends the
upstream worker in two ways:

1. After the base ``init_model`` builds the FSDP actor (and optional reference)
   modules, the standard ``DataParallelPPOActor`` instances are replaced with
   :class:`ComposerDataParallelPPOActor` so that custom flow-aware logic
   (loss composition, hidden-state surfacing, info-NCE) is used.
2. ``compute_log_prob`` is overridden so the worker forwards the hidden states
   produced by the actor's ``compute_log_prob`` into the returned ``DataProto``
   under both ``hidden_states`` and ``response_hidden_states`` keys, which the
   composer flows (DRA-GRPO, etc.) require.

Step 7 only *defines* the subclass; it is wired into ``ComposerRayPPOTrainer``
via :class:`ComposerTaskRunner` (see ``entrypoint.py``).
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Optional

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    from verl import DataProto
    from verl.single_controller.base.decorator import (
        Dispatch,
        make_nd_compute_dataproto_dispatch_fn,
        register,
    )
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    DataProto = None
    Dispatch = None
    make_nd_compute_dataproto_dispatch_fn = None
    register = None
    omega_conf_to_dataclass = None
    fsdp_version = None
    load_fsdp_model_to_gpu = None
    offload_fsdp_model_to_cpu = None
    DistProfiler = None
    log_gpu_memory_usage = None
    ActorRolloutRefWorker = object  # type: ignore[assignment]

from .composer_dp_actor import ComposerDataParallelPPOActor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ComposerActorRolloutRefWorker(ActorRolloutRefWorker):
    """veRL ``ActorRolloutRefWorker`` that uses ``ComposerDataParallelPPOActor``.

    The class is intentionally thin: it lets the upstream ``init_model`` do all
    of the heavy FSDP / rollout / checkpoint plumbing, then swaps the actor (and
    optional reference) policy objects to the composer subclass. Both replacements
    re-use the FSDP-wrapped modules and optimizer instances created by the
    parent, so the checkpoint manager bound earlier in ``init_model`` continues
    to reference the same parameters.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):  # type: ignore[override]
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = ComposerDataParallelPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        # Standalone reference policy (only created when not using LoRA-shared ref).
        if self._is_ref and not self._is_lora and getattr(self, "ref_module_fsdp", None) is not None:
            self.ref_policy = ComposerDataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data):  # type: ignore[override]
        """Recompute log-probs and surface hidden states for composer flows."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()

        # Align with veRL defaults for log-prob recompute.
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output_tuple = self.actor.compute_log_prob(data=data, calculate_entropy=True)

            if not isinstance(output_tuple, (tuple, list)):
                raise TypeError(
                    "Actor compute_log_prob must return a tuple: "
                    "(log_probs, entropys) or (log_probs, entropys, hidden_states)."
                )

            if len(output_tuple) == 3:
                old_log_probs, entropys, hidden_states = output_tuple
            elif len(output_tuple) == 2:
                raise ValueError(
                    "compute_log_prob returned no hidden states. "
                    "DRA-GRPO requires hidden states."
                )
            else:
                raise ValueError(
                    "Unexpected compute_log_prob output length. "
                    f"Expected 2 or 3 values, got {len(output_tuple)}."
                )

            if hidden_states is None:
                raise ValueError(
                    "compute_log_prob returned hidden_states=None. DRA-GRPO requires hidden states."
                )

            tensors = {
                "old_log_probs": old_log_probs,
                "entropys": entropys,
                "hidden_states": hidden_states,
                "response_hidden_states": hidden_states,
            }

            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output


__all__ = ["ComposerActorRolloutRefWorker"]
