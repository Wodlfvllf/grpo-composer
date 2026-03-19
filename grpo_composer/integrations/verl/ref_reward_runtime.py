"""Runtime helpers for PVPO/GAPO reference reward construction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import ray
import torch


def has_reference_rewards(data: Any) -> bool:
    if hasattr(data, "batch") and "reference_rewards" in data.batch:
        return True
    non_tensor = getattr(data, "non_tensor_batch", None)
    return isinstance(non_tensor, Mapping) and "reference_rewards" in non_tensor


def set_do_sample_flag(data: Any, do_sample: bool) -> None:
    meta_info = getattr(data, "meta_info", None)
    if meta_info is None:
        try:
            data.meta_info = {}
            meta_info = data.meta_info
        except Exception:
            return
    if isinstance(meta_info, dict):
        meta_info["do_sample"] = do_sample
        return
    setter = getattr(meta_info, "__setitem__", None)
    if callable(setter):
        try:
            setter("do_sample", do_sample)
        except Exception:
            pass


def _generate_reference_rollout_output(trainer: Any, ref_batch: Any, ref_reward_source: str) -> Any:
    ref_wg = getattr(trainer, "ref_policy_wg", None)
    fallback_to_actor = False
    last_error: Exception | None = None

    if ref_wg is not None:
        try:
            return ref_wg.generate_sequences(ref_batch)
        except Exception as exc:
            last_error = exc
            if "rollout is not registered in ActorRolloutRefWorker" not in str(exc):
                raise
            fallback_to_actor = ref_reward_source in ("auto", "actor_rollout")
    else:
        fallback_to_actor = ref_reward_source in ("auto", "actor_rollout")

    if not fallback_to_actor and ref_wg is not None:
        raise RuntimeError(
            "PVPO reference rollouts are unavailable on this veRL setup: "
            "the `ref` worker does not register the `rollout` mesh in `verl==0.6.x`. "
            "Provide `reference_rewards` in the batch, or set "
            "`composer.reference_reward_source=actor_rollout` for an approximate fallback."
        ) from last_error
    if not fallback_to_actor and ref_wg is None:
        raise RuntimeError(
            "PVPO reference rollouts requested but no `ref_policy_wg` is available. "
            "Provide `reference_rewards` in the batch, or set "
            "`composer.reference_reward_source=actor_rollout` for an approximate fallback."
        )

    rollout_manager = getattr(trainer, "async_rollout_manager", None)
    if rollout_manager is None:
        rollout_manager = getattr(trainer, "actor_rollout_wg", None)
    if rollout_manager is None:
        raise RuntimeError(
            "Cannot compute reference rewards: no rollout manager is available for fallback generation."
        ) from last_error

    if fallback_to_actor and not getattr(trainer, "_pvpo_actor_rollout_fallback_warned", False):
        print(
            "[grpo_composer] PVPO fallback active: using actor rollout manager for reference rewards "
            "(approximation, not true reference-policy rollouts)."
        )
        trainer._pvpo_actor_rollout_fallback_warned = True

    return rollout_manager.generate_sequences(ref_batch)


def _build_reference_eval_batch(ref_batch: Any, ref_output: Any, debug: bool) -> Any:
    try:
        return ref_batch.union(ref_output)
    except AssertionError as exc:
        if debug:
            print(f"[grpo_composer-debug] ref_batch.union(ref_output) failed: {exc}")
        eval_batch = ref_output
        if hasattr(ref_batch, "non_tensor_batch") and hasattr(eval_batch, "non_tensor_batch"):
            for k, v in ref_batch.non_tensor_batch.items():
                if k not in eval_batch.non_tensor_batch:
                    eval_batch.non_tensor_batch[k] = v
        return eval_batch


def _compute_reference_reward_tensor(trainer: Any, ref_eval_batch: Any):
    if (
        getattr(trainer.config, "reward_model", None) is not None
        and getattr(trainer.config.reward_model, "launch_reward_fn_async", False)
    ):
        try:
            from verl.trainer.ppo.reward import compute_reward_async
        except ImportError:
            from verl.trainer.ppo.core_algos import compute_reward_async
        future_reward = compute_reward_async.remote(
            data=ref_eval_batch, config=trainer.config, tokenizer=trainer.tokenizer
        )
        ref_reward_tensor, _ = ray.get(future_reward)
    else:
        if hasattr(trainer, "_compute_or_extract_reward"):
            ref_reward_tensor, _ = trainer._compute_or_extract_reward(
                ref_eval_batch, reward_fn=trainer.reward_fn, return_dict=False
            )
        else:
            try:
                from verl.trainer.ppo.reward import compute_reward
            except ImportError:
                from verl.trainer.ppo.core_algos import compute_reward
            ref_reward_tensor, _ = compute_reward(ref_eval_batch, trainer.reward_fn)
    return ref_reward_tensor


def ensure_reference_rewards(
    trainer: Any,
    data: Any,
    *,
    composer_cfg: Mapping[str, Any] | dict[str, Any],
    ref_reward_source: str,
    debug: bool = False,
) -> None:
    if has_reference_rewards(data):
        return

    ref_batch = copy.deepcopy(data)
    do_sample = bool(
        composer_cfg.get(
            "reference_rollout_do_sample",
            getattr(trainer.config, "reference_rollout_do_sample", False),
        )
    )
    set_do_sample_flag(ref_batch, do_sample)

    ref_output = _generate_reference_rollout_output(trainer, ref_batch, ref_reward_source)
    ref_eval_batch = _build_reference_eval_batch(ref_batch, ref_output, debug)
    ref_reward_tensor = _compute_reference_reward_tensor(trainer, ref_eval_batch)

    expected_bs = None
    if hasattr(data, "batch") and "responses" in data.batch:
        try:
            expected_bs = int(data.batch["responses"].shape[0])
        except Exception:
            expected_bs = None

    if isinstance(ref_reward_tensor, torch.Tensor):
        if expected_bs is not None and int(ref_reward_tensor.shape[0]) != expected_bs:
            raise RuntimeError(
                "Reference reward batch size mismatch: "
                f"expected {expected_bs}, got {tuple(ref_reward_tensor.shape)}"
            )
        data.non_tensor_batch["reference_rewards"] = ref_reward_tensor.cpu().numpy()
    else:
        ref_np = np.asarray(ref_reward_tensor)
        if expected_bs is not None and int(ref_np.shape[0]) != expected_bs:
            raise RuntimeError(
                "Reference reward batch size mismatch: "
                f"expected {expected_bs}, got {ref_np.shape}"
            )
        data.non_tensor_batch["reference_rewards"] = ref_np

    if debug:
        shape = data.non_tensor_batch["reference_rewards"].shape
        print(f"[grpo_composer-debug] Added reference_rewards shape={shape}")
