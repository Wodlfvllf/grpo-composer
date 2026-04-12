"""Worker-side patching for veRL FSDP actor `compute_log_prob`."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Optional

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    from verl import DataProto
    from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
    from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
    from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    DataProto = None
    ActorRolloutRefWorker = None
    register = None
    make_nd_compute_dataproto_dispatch_fn = None
    DistProfiler = None
    fsdp_version = None
    load_fsdp_model_to_gpu = None
    offload_fsdp_model_to_cpu = None
    log_gpu_memory_usage = None


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ORIGINAL_FSDP_COMPUTE_LOG_PROB = None


def _patch_fsdp_compute_log_probs() -> None:
    """Patch FSDP worker to return hidden states alongside old log-probs."""

    global _ORIGINAL_FSDP_COMPUTE_LOG_PROB

    if ActorRolloutRefWorker is None or DataProto is None or register is None:
        return
    if _ORIGINAL_FSDP_COMPUTE_LOG_PROB is not None:
        return

    _ORIGINAL_FSDP_COMPUTE_LOG_PROB = ActorRolloutRefWorker.compute_log_prob

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def _patched_compute_log_prob(self, data: DataProto):
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
            }
            tensors["hidden_states"] = hidden_states
            tensors["response_hidden_states"] = hidden_states

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

    ActorRolloutRefWorker.compute_log_prob = _patched_compute_log_prob
