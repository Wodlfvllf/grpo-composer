"""Composer subclass of veRL's :class:`DataParallelPPOActor`.

Step 6 of the monkey-patch removal refactor (see commit history).

The three methods previously installed as class-level monkey-patches by
``patch_dp_actor._patch_dp_actor_update_policy`` and
``patch_dp_actor._patch_dp_actor_forward_microbatch_compute_log_prob`` are
now real overrides on this subclass. The subclass is defined alongside the
existing patch so this commit is behavior-preserving on its own; Steps 7
and 8 will wire ``ComposerActorRolloutRefWorker`` to instantiate this
class and remove ``patch_dp_actor`` entirely.

When ``loss_mode != "composer"``, ``update_policy`` falls through to
``super().update_policy(data)`` instead of the patch's
``_ORIGINAL_DP_ACTOR_UPDATE_POLICY(self, data)`` global.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    from verl import DataProto
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
    from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
    from verl.utils.device import get_device_id
    from verl.utils.profiler import GPUMemoryLogger
    from verl.utils.py_functional import append_to_dict
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
    from verl.utils.torch_functional import logprobs_from_logits
    from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    import verl.utils.torch_functional as verl_F
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    DataParallelPPOActor = None  # type: ignore[assignment]
    DataProto = None  # type: ignore[assignment]
    GPUMemoryLogger = None  # type: ignore[assignment]


# Reuse module-level helpers from the patch module to keep a single source
# of truth. Step 10 will move these to integrations/verl/utils.py.
from .patch_dp_actor import (
    _cfg_get,
    _extract_rollout_n,
    _shape_debug,
    _strict_validation_enabled,
)


if DataParallelPPOActor is None:
    class ComposerDataParallelPPOActor:  # type: ignore[no-redef]
        """Stub used when verl is unavailable so the module imports cleanly."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ComposerDataParallelPPOActor requires `verl` to be installed. "
                f"Original import error: {_VERL_IMPORT_ERROR!r}"
            )
else:

    class ComposerDataParallelPPOActor(DataParallelPPOActor):
        """Subclass of veRL's :class:`DataParallelPPOActor` for grpo_composer.

        Behavior matches ``patch_dp_actor`` exactly; this class only relocates
        the three patched methods into proper overrides.
        """

        # ------------------------------------------------------------------
        # update_policy
        # ------------------------------------------------------------------
        def update_policy(self, data):  # type: ignore[override]
            from .loss_context import (
                clear_composer_batch_context,
                set_composer_batch_context,
                set_composer_config,
            )

            clear_batch_context = clear_composer_batch_context

            try:
                meta_info = getattr(data, "meta_info", None)
                composer_cfg = None
                if isinstance(meta_info, dict):
                    composer_cfg = meta_info.get("composer_config")
                else:
                    getter = getattr(meta_info, "get", None)
                    if callable(getter):
                        composer_cfg = getter("composer_config", None)

                if isinstance(composer_cfg, Mapping):
                    composer_cfg = dict(composer_cfg)
                elif composer_cfg is None:
                    composer_cfg_json = None
                    if isinstance(meta_info, dict):
                        composer_cfg_json = meta_info.get("composer_config_json")
                    else:
                        getter = getattr(meta_info, "get", None)
                        if callable(getter):
                            composer_cfg_json = getter("composer_config_json", None)

                    if isinstance(composer_cfg_json, str) and composer_cfg_json:
                        try:
                            parsed = json.loads(composer_cfg_json)
                            if isinstance(parsed, dict):
                                composer_cfg = parsed
                        except Exception:
                            composer_cfg = None

                if composer_cfg is None:
                    primitive_cfg: dict[str, Any] = {}
                    lookup_keys = ("clip_mode", "agg_mode", "regularizer", "reg_coef")
                    if isinstance(meta_info, dict):
                        for k in lookup_keys:
                            meta_key = f"composer_{k}"
                            if meta_key in meta_info and meta_info[meta_key] is not None:
                                primitive_cfg[k] = meta_info[meta_key]
                    else:
                        getter = getattr(meta_info, "get", None)
                        if callable(getter):
                            for k in lookup_keys:
                                meta_key = f"composer_{k}"
                                v = getter(meta_key, None)
                                if v is not None:
                                    primitive_cfg[k] = v
                    if primitive_cfg:
                        composer_cfg = primitive_cfg

                if composer_cfg is None:
                    raw = os.environ.get("GRPO_COMPOSER_CONFIG")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                composer_cfg = parsed
                        except Exception:
                            pass

                if isinstance(composer_cfg, dict) and composer_cfg:
                    set_composer_config(composer_cfg)
                    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                        agg = composer_cfg.get("agg_mode", "<missing>")
                        clip = composer_cfg.get("clip_mode", "<missing>")
                        reg = composer_cfg.get("regularizer", "<missing>")
                        print(
                            f"[composer-debug] Bound worker composer config: "
                            f"clip={clip}, agg={agg}, regularizer={reg}"
                        )
                elif os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                    keys = []
                    if isinstance(meta_info, dict):
                        keys = list(meta_info.keys())
                    print(
                        "[composer-debug] Missing composer config in worker update_policy. "
                        f"meta_info_type={type(meta_info)}, meta_info_keys={keys}"
                    )

                def _read_key(container: Any, key: str):
                    if container is None:
                        return None
                    if isinstance(container, dict):
                        return container.get(key)
                    getter = getattr(container, "get", None)
                    if callable(getter):
                        try:
                            return getter(key, None)
                        except TypeError:
                            try:
                                return getter(key)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    try:
                        keys = getattr(container, "keys", None)
                        if callable(keys) and key in keys():
                            return container[key]
                    except Exception:
                        pass
                    return None

                batch_context: dict[str, Any] = {}
                non_tensor_batch = getattr(data, "non_tensor_batch", None)
                tensor_batch = getattr(data, "batch", None)

                uid = _read_key(non_tensor_batch, "uid")
                if uid is None:
                    uid = _read_key(tensor_batch, "uid")

                # ── uid fixup (worker side) ────────────────────────────────
                if uid is not None:
                    _uid_arr = np.asarray(uid)
                    _total = _uid_arr.shape[0]
                    _unique = len(set(_uid_arr.tolist()))
                    _n_repeat = None
                    for _mi_key in ("rollout_n", "n"):
                        if _n_repeat is not None:
                            break
                        if isinstance(meta_info, dict):
                            _n_repeat = meta_info.get(_mi_key)
                        else:
                            _getter = getattr(meta_info, "get", None)
                            if callable(_getter):
                                try:
                                    _n_repeat = _getter(_mi_key, None)
                                except Exception:
                                    pass
                    if _n_repeat is None and isinstance(composer_cfg, dict):
                        _n_repeat = composer_cfg.get("rollout_n")
                    if _n_repeat is None:
                        _cfg_env = os.environ.get("GRPO_COMPOSER_ROLLOUT_N")
                        if _cfg_env is not None:
                            try:
                                _n_repeat = int(_cfg_env)
                            except ValueError:
                                pass
                    if _n_repeat is not None:
                        _n_repeat = int(_n_repeat)
                    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                        print(
                            f"[composer-debug] worker uid fixup check: total={_total} "
                            f"unique={_unique} n_repeat={_n_repeat}"
                        )
                    if (
                        _n_repeat is not None
                        and _n_repeat > 1
                        and _unique == _total
                        and _total % _n_repeat == 0
                    ):
                        _fixed = np.array(
                            [_uid_arr[i * _n_repeat] for i in range(_total // _n_repeat) for _ in range(_n_repeat)],
                            dtype=_uid_arr.dtype,
                        )
                        uid = _fixed
                        if non_tensor_batch is not None:
                            try:
                                non_tensor_batch["uid"] = _fixed
                                data.non_tensor_batch["uid"] = _fixed
                            except Exception:
                                pass
                        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                            print(
                                f"[composer-debug] worker uid fixup: {_total} rows had {_unique} unique uids "
                                f"→ reconstructed {len(set(_fixed.tolist()))} unique uids (n={_n_repeat})"
                            )

                if uid is not None:
                    batch_context["composer_uid"] = uid

                sequence_rewards = None
                token_level_rewards = _read_key(tensor_batch, "token_level_rewards")
                response_mask = _read_key(tensor_batch, "response_mask")
                if (
                    isinstance(token_level_rewards, torch.Tensor)
                    and isinstance(response_mask, torch.Tensor)
                    and token_level_rewards.shape == response_mask.shape
                ):
                    sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)

                if sequence_rewards is None:
                    sequence_rewards = _read_key(tensor_batch, "composer_sequence_rewards")
                if sequence_rewards is None:
                    sequence_rewards = _read_key(tensor_batch, "sequence_rewards")
                if sequence_rewards is None:
                    sequence_rewards = _read_key(non_tensor_batch, "composer_sequence_rewards")
                if sequence_rewards is None:
                    sequence_rewards = _read_key(non_tensor_batch, "sequence_rewards")

                if isinstance(sequence_rewards, np.ndarray):
                    sequence_rewards = torch.from_numpy(sequence_rewards)
                elif sequence_rewards is not None and not isinstance(sequence_rewards, torch.Tensor):
                    try:
                        sequence_rewards = torch.as_tensor(sequence_rewards)
                    except Exception:
                        sequence_rewards = None

                if isinstance(sequence_rewards, torch.Tensor):
                    if sequence_rewards.ndim == 2:
                        if (
                            isinstance(response_mask, torch.Tensor)
                            and sequence_rewards.shape == response_mask.shape
                        ):
                            sequence_rewards = (sequence_rewards * response_mask).sum(dim=-1)
                        else:
                            sequence_rewards = None
                    elif sequence_rewards.ndim != 1:
                        sequence_rewards = None

                agg_mode = None
                if isinstance(composer_cfg, Mapping):
                    agg_mode = composer_cfg.get("agg_mode")

                # λ-GRPO learnable aggregation module ─────────────────────
                if agg_mode == "group_learnable":
                    from grpo_composer.core.aggregation.group_learnable import GroupLearnableAggregation

                    lambda_init = float(_cfg_get(self.config, "lambda_init", 0.0))
                    lambda_r = float(_cfg_get(self.config, "lambda_r", 0.1111))
                    lambda_learnable = bool(_cfg_get(self.config, "lambda_learnable", False))
                    lambda_lr = float(_cfg_get(self.config, "lambda_lr", 0.1))

                    module_spec = (lambda_init, lambda_r, lambda_learnable, lambda_lr)
                    module = getattr(self, "_composer_group_learnable_module", None)
                    if module is None or getattr(self, "_composer_group_learnable_spec", None) != module_spec:
                        module = GroupLearnableAggregation(
                            lambda_=lambda_init,
                            r=lambda_r,
                            learnable=lambda_learnable,
                        )
                        setattr(self, "_composer_group_learnable_module", module)
                        setattr(self, "_composer_group_learnable_spec", module_spec)
                        setattr(self, "_composer_group_learnable_opt_registered", False)

                    if lambda_learnable and isinstance(getattr(module, "lambda_", None), torch.nn.Parameter):
                        try:
                            actor_device = next(self.actor_module.parameters()).device
                        except Exception:
                            actor_device = module.lambda_.device

                        if module.lambda_.device != actor_device:
                            module.lambda_ = torch.nn.Parameter(module.lambda_.detach().to(actor_device))
                            setattr(self, "_composer_group_learnable_opt_registered", False)

                        already_in_optimizer = False
                        for param_group in self.actor_optimizer.param_groups:
                            for param in param_group.get("params", []):
                                if param is module.lambda_:
                                    already_in_optimizer = True
                                    break
                            if already_in_optimizer:
                                break

                        if already_in_optimizer:
                            setattr(self, "_composer_group_learnable_opt_registered", True)

                        if not bool(getattr(self, "_composer_group_learnable_opt_registered", False)):
                            self.actor_optimizer.add_param_group(
                                {
                                    "params": [module.lambda_],
                                    "lr": lambda_lr,
                                    "weight_decay": 0.0,
                                }
                            )
                            setattr(self, "_composer_group_learnable_opt_registered", True)
                            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                                print(
                                    "[composer-debug] Registered learnable lambda with actor optimizer: "
                                    f"lambda_init={lambda_init} lambda_r={lambda_r} lambda_lr={lambda_lr}"
                                )

                    batch_context["composer_group_learnable_module"] = module

                # DARO learnable aggregation module ───────────────────────
                if agg_mode == "difficulty_weighted":
                    from grpo_composer.core.aggregation.difficulty_weighted import DifficultyWeightedAggregation

                    num_bins = int(_cfg_get(composer_cfg, "difficulty_bins", 10))
                    weight_c = float(_cfg_get(composer_cfg, "difficulty_weight_c", 1.0))
                    learnable = bool(_cfg_get(composer_cfg, "difficulty_weight_learnable", True))
                    init_weight = float(_cfg_get(composer_cfg, "difficulty_weight_init", 1.0))

                    module_spec = (num_bins, weight_c, learnable, init_weight)
                    module = getattr(self, "_composer_difficulty_agg_module", None)
                    if module is None or getattr(self, "_composer_difficulty_agg_spec", None) != module_spec:
                        module = DifficultyWeightedAggregation(
                            num_bins=num_bins,
                            weight_c=weight_c,
                            learnable=learnable,
                            init_weight=init_weight,
                        )
                        setattr(self, "_composer_difficulty_agg_module", module)
                        setattr(self, "_composer_difficulty_agg_spec", module_spec)
                        setattr(self, "_composer_difficulty_agg_opt_registered", False)

                    if learnable and getattr(module, "weight_params", None) is not None:
                        try:
                            actor_device = next(self.actor_module.parameters()).device
                        except Exception:
                            actor_device = module.weight_params.device

                        if module.weight_params.device != actor_device:
                            module.weight_params = torch.nn.Parameter(
                                module.weight_params.detach().to(actor_device)
                            )
                            setattr(self, "_composer_difficulty_agg_opt_registered", False)

                        already_in_optimizer = False
                        for param_group in self.actor_optimizer.param_groups:
                            for param in param_group.get("params", []):
                                if param is module.weight_params:
                                    already_in_optimizer = True
                                    break
                            if already_in_optimizer:
                                break

                        if already_in_optimizer:
                            setattr(self, "_composer_difficulty_agg_opt_registered", True)

                        if not bool(getattr(self, "_composer_difficulty_agg_opt_registered", False)):
                            self.actor_optimizer.add_param_group({"params": [module.weight_params]})
                            setattr(self, "_composer_difficulty_agg_opt_registered", True)
                            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                                print(
                                    "[composer-debug] Registered learnable DARO weights with actor optimizer: "
                                    f"num_bins={num_bins}"
                                )

                    batch_context["composer_difficulty_agg_module"] = module

                if agg_mode == "difficulty_weighted":
                    if uid is None:
                        raise ValueError(
                            "Composer validation failed in worker update_policy: "
                            "agg_mode='difficulty_weighted' requires uid for prompt grouping."
                        )
                    if not (
                        isinstance(token_level_rewards, torch.Tensor)
                        and isinstance(response_mask, torch.Tensor)
                        and token_level_rewards.shape == response_mask.shape
                    ):
                        candidates = {
                            "token_level_rewards": _shape_debug(_read_key(tensor_batch, "token_level_rewards")),
                            "response_mask": _shape_debug(_read_key(tensor_batch, "response_mask")),
                        }
                        raise ValueError(
                            "Composer validation failed in worker update_policy: "
                            "agg_mode='difficulty_weighted' requires token_level_rewards aligned with response_mask. "
                            f"Observed candidates: {candidates}"
                        )

                if isinstance(sequence_rewards, torch.Tensor):
                    batch_context["composer_sequence_rewards"] = sequence_rewards

                rollout_n = _extract_rollout_n(meta_info)
                if rollout_n is not None:
                    batch_context["rollout_n"] = rollout_n

                if batch_context:
                    set_composer_batch_context(batch_context)
                    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                        print(
                            "[composer-debug] Bound worker batch context: "
                            f"keys={sorted(batch_context.keys())}"
                        )
                else:
                    clear_composer_batch_context()
            except Exception as exc:
                if _strict_validation_enabled():
                    raise RuntimeError(
                        "Composer worker preflight validation failed before actor update. "
                        f"Root cause: {type(exc).__name__}: {exc}"
                    ) from exc

            try:
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                if loss_mode != "composer":
                    return super().update_policy(data)

                # Inlined veRL update_policy with minimal additions for composer loss.
                self.actor_module.train()
                temperature = data.meta_info["temperature"]

                required_select_keys = [
                    "responses",
                    "response_mask",
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "old_log_probs",
                    "advantages",
                ]
                select_keys = list(required_select_keys)
                for key in ("hidden_states", "response_hidden_states"):
                    if key in data.batch.keys():
                        select_keys.append(key)
                if self.config.use_kl_loss or "ref_log_prob" in data.batch.keys():
                    select_keys.append("ref_log_prob")
                if "rollout_is_weights" in data.batch.keys():
                    select_keys.append("rollout_is_weights")
                if "rollout_log_probs" in data.batch.keys():
                    select_keys.append("rollout_log_probs")
                if "token_level_rewards" in data.batch.keys():
                    select_keys.append("token_level_rewards")

                non_tensor_select_keys = []
                if "multi_modal_inputs" in data.non_tensor_batch.keys():
                    non_tensor_select_keys.append("multi_modal_inputs")
                if "uid" in data.non_tensor_batch.keys():
                    non_tensor_select_keys.append("uid")

                data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

                mini_batches = data.split(self.config.ppo_mini_batch_size)
                on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

                meta_info = getattr(data, "meta_info", None)
                rollout_n = _extract_rollout_n(meta_info)

                metrics = {
                    "actor/pg_loss": 0.0,
                    "actor/kl_loss": 0.0,
                }
                agg_mode = _cfg_get(composer_cfg, "agg_mode", "token_mean")
                daro_enabled = agg_mode == "difficulty_weighted"
                daro_num_bins = int(_cfg_get(composer_cfg, "difficulty_bins", 10))
                daro_eps = float(_cfg_get(composer_cfg, "difficulty_epsilon", 1e-8))
                daro_module = getattr(self, "_composer_difficulty_agg_module", None)
                if daro_module is not None:
                    daro_num_bins = int(getattr(daro_module, "num_bins", daro_num_bins))
                    daro_eps = float(getattr(daro_module, "epsilon", daro_eps))

                for _ in range(self.config.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(mini_batches):
                        if daro_enabled and os.environ.get("GRPO_COMPOSER_DEBUG") == "1" and batch_idx == 0:
                            data_uid = None
                            if "uid" in data.non_tensor_batch:
                                data_uid = data.non_tensor_batch["uid"]
                            elif "uid" in data.batch.keys():
                                data_uid = data.batch["uid"]

                            if data_uid is not None:
                                if isinstance(data_uid, torch.Tensor):
                                    data_uid_arr = data_uid.detach().cpu().numpy()
                                else:
                                    data_uid_arr = np.asarray(data_uid)
                                uid_counts = defaultdict(int)
                                for uid_key in data_uid_arr.tolist():
                                    uid_counts[uid_key] += 1
                                count_hist = defaultdict(int)
                                for cnt in uid_counts.values():
                                    count_hist[int(cnt)] += 1
                                hist_sorted = {k: count_hist[k] for k in sorted(count_hist.keys())}
                                print(
                                    "[composer-debug][daro] actor-batch uid multiplicity: "
                                    f"rows={len(data_uid_arr)} unique_uids={len(uid_counts)} "
                                    f"hist={hist_sorted}"
                                )

                        mini_uid_to_bin: dict[Any, int] = {}
                        mini_uid_to_inv_group_tokens: dict[Any, float] = {}
                        mini_active_mu_ids: list[int] = []
                        mini_active_mu_ids_tensor: torch.Tensor | None = None
                        micro_debug_idx = 0
                        if daro_enabled:
                            mini_mask = mini_batch.batch.get("response_mask", None)
                            mini_rewards = mini_batch.batch.get("token_level_rewards", None)

                            mini_uid = None
                            if "uid" in mini_batch.non_tensor_batch:
                                mini_uid = mini_batch.non_tensor_batch["uid"]
                            elif "uid" in mini_batch.batch.keys():
                                mini_uid = mini_batch.batch["uid"]

                            if mini_uid is None:
                                raise ValueError(
                                    "DARO requires uid in mini_batch for prompt grouping."
                                )
                            if not isinstance(mini_mask, torch.Tensor):
                                raise ValueError(
                                    "DARO requires mini_batch.response_mask as torch.Tensor."
                                )
                            if not isinstance(mini_rewards, torch.Tensor):
                                raise ValueError(
                                    "DARO requires mini_batch.token_level_rewards as torch.Tensor."
                                )
                            if mini_rewards.shape != mini_mask.shape:
                                raise ValueError(
                                    "DARO requires token_level_rewards shape to match response_mask, got "
                                    f"{tuple(mini_rewards.shape)} vs {tuple(mini_mask.shape)}"
                                )

                            mini_seq_rewards = (
                                mini_rewards * mini_mask.to(dtype=mini_rewards.dtype)
                            ).sum(dim=-1).detach().cpu()
                            mini_token_counts = mini_mask.sum(dim=-1).to(dtype=torch.float32).detach().cpu()

                            if isinstance(mini_uid, torch.Tensor):
                                mini_uid_arr = mini_uid.detach().cpu().numpy()
                            else:
                                mini_uid_arr = np.asarray(mini_uid)

                            if mini_uid_arr.ndim != 1 or mini_uid_arr.shape[0] != int(mini_seq_rewards.shape[0]):
                                raise ValueError(
                                    "DARO requires uid shape [B_mini], got "
                                    f"{tuple(mini_uid_arr.shape)} for B_mini={int(mini_seq_rewards.shape[0])}"
                                )
                            mini_uid_list = mini_uid_arr.tolist()

                            groups: dict[Any, list[int]] = defaultdict(list)
                            for idx, uid_key in enumerate(mini_uid_list):
                                groups[uid_key].append(idx)

                            bin_token_counts: dict[int, float] = defaultdict(float)
                            for uid_key, indices in groups.items():
                                idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=mini_seq_rewards.device)
                                prompt_rewards = mini_seq_rewards.index_select(0, idx_tensor)
                                mu = float((prompt_rewards > 0).float().mean().item())
                                if mu <= 0.0 or mu >= 1.0:
                                    mini_uid_to_bin[uid_key] = -1
                                    continue
                                mu_bin = min(int(mu * daro_num_bins), daro_num_bins - 1)
                                mini_uid_to_bin[uid_key] = mu_bin
                                prompt_tokens = float(mini_token_counts.index_select(0, idx_tensor).sum().item())
                                bin_token_counts[mu_bin] += prompt_tokens

                            mini_active_mu_ids = sorted(
                                [bin_id for bin_id, token_count in bin_token_counts.items() if token_count > 0.0]
                            )
                            for uid_key, mu_bin in mini_uid_to_bin.items():
                                if mu_bin < 0:
                                    mini_uid_to_inv_group_tokens[uid_key] = 0.0
                                    continue
                                total_tokens = float(bin_token_counts.get(mu_bin, 0.0))
                                if total_tokens <= 0.0:
                                    mini_uid_to_inv_group_tokens[uid_key] = 0.0
                                else:
                                    mini_uid_to_inv_group_tokens[uid_key] = 1.0 / (total_tokens + daro_eps)

                            mini_mu_id_row = torch.tensor(
                                [int(mini_uid_to_bin[u]) for u in mini_uid_list],
                                device=mini_mask.device,
                                dtype=torch.long,
                            )
                            mini_inv_group_tokens_row = torch.tensor(
                                [float(mini_uid_to_inv_group_tokens[u]) for u in mini_uid_list],
                                device=mini_mask.device,
                                dtype=mini_rewards.dtype,
                            )
                            mini_active_mu_ids_tensor = torch.tensor(
                                mini_active_mu_ids,
                                device=mini_mask.device,
                                dtype=torch.long,
                            )

                            mini_batch.batch["daro_mu_id_row"] = mini_mu_id_row
                            mini_batch.batch["daro_inv_group_tokens_row"] = mini_inv_group_tokens_row

                            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                                bin_token_counts_debug = {
                                    int(k): float(v) for k, v in sorted(bin_token_counts.items(), key=lambda kv: kv[0])
                                }
                                valid_rows = mini_mu_id_row[mini_mu_id_row >= 0]
                                if valid_rows.numel() > 0:
                                    bincount = torch.bincount(valid_rows, minlength=daro_num_bins).detach().cpu().tolist()
                                else:
                                    bincount = [0] * daro_num_bins
                                preview_n = min(8, int(mini_mu_id_row.shape[0]))
                                print(
                                    "[composer-debug][daro] mini_batch context: "
                                    f"prompts={len(groups)} rows={len(mini_uid_list)} "
                                    f"active_bins={mini_active_mu_ids}"
                                )
                                print(
                                    "[composer-debug][daro] mini_batch math: "
                                    f"N_mu={bin_token_counts_debug} "
                                    f"row_bin_counts={bincount} "
                                    f"mu_id_row[:{preview_n}]={mini_mu_id_row[:preview_n].detach().cpu().tolist()} "
                                    f"inv_N_row[:{preview_n}]={mini_inv_group_tokens_row[:preview_n].detach().cpu().tolist()}"
                                )

                        if self.config.use_dynamic_bsz:
                            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                            micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                        else:
                            self.gradient_accumulation = (
                                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                            )
                            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                        self.actor_optimizer.zero_grad()

                        for micro_batch in micro_batches:
                            micro_batch = micro_batch.to(get_device_id())
                            micro_batch_metrics = {}
                            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                            response_mask = model_inputs["response_mask"]
                            old_log_prob = model_inputs["old_log_probs"]
                            advantages = model_inputs["advantages"]

                            hidden_states = model_inputs.get("response_hidden_states", None)
                            if hidden_states is None:
                                hidden_states = model_inputs.get("hidden_states", None)

                            if os.environ.get("GRPO_COMPOSER_DEBUG_UID_MICROBATCH", "0") == "1":
                                max_uid_logs = 8
                                try:
                                    max_uid_logs = int(os.environ.get("GRPO_COMPOSER_DEBUG_UID_MAX_MICROBATCH", "8"))
                                except Exception:
                                    pass

                                if micro_debug_idx < max_uid_logs:
                                    uid_value = model_inputs.get("uid", None)
                                    strict_uid = os.environ.get("GRPO_COMPOSER_DEBUG_UID_STRICT", "0") == "1"
                                    uid_ok = True

                                    if uid_value is None:
                                        uid_ok = False
                                        print(
                                            "[composer-debug][uid] micro_batch check: "
                                            f"mini_idx={batch_idx} micro_idx={micro_debug_idx} "
                                            "uid missing"
                                        )
                                    else:
                                        if isinstance(uid_value, torch.Tensor):
                                            uid_arr = uid_value.detach().cpu().numpy()
                                        else:
                                            uid_arr = np.asarray(uid_value)

                                        b_micro = int(response_mask.shape[0]) if isinstance(response_mask, torch.Tensor) else None
                                        if uid_arr.ndim != 1 or b_micro is None or uid_arr.shape[0] != b_micro:
                                            uid_ok = False
                                            print(
                                                "[composer-debug][uid] micro_batch check: "
                                                f"mini_idx={batch_idx} micro_idx={micro_debug_idx} "
                                                f"uid_shape={tuple(uid_arr.shape)} B_micro={b_micro}"
                                            )
                                        else:
                                            uid_list = uid_arr.tolist()
                                            uid_counts: dict[Any, int] = defaultdict(int)
                                            for key in uid_list:
                                                uid_counts[key] += 1
                                            count_hist: dict[int, int] = defaultdict(int)
                                            for cnt in uid_counts.values():
                                                count_hist[int(cnt)] += 1

                                            run_lengths: list[tuple[Any, int]] = []
                                            prev_key: Any = None
                                            run_count = 0
                                            for key in uid_list:
                                                if run_count == 0 or key != prev_key:
                                                    if run_count > 0:
                                                        run_lengths.append((prev_key, run_count))
                                                    prev_key = key
                                                    run_count = 1
                                                else:
                                                    run_count += 1
                                            if run_count > 0:
                                                run_lengths.append((prev_key, run_count))

                                            expected = int(rollout_n) if rollout_n is not None else None
                                            expected_source = "meta"
                                            count_ok = None
                                            order_ok = None
                                            prompt_count_expected = None
                                            if expected is None:
                                                unique_counts = sorted(set(int(c) for c in uid_counts.values()))
                                                if len(unique_counts) == 1 and unique_counts[0] > 0:
                                                    inferred = unique_counts[0]
                                                    if b_micro % inferred == 0:
                                                        expected = inferred
                                                        expected_source = "inferred"
                                            if expected is not None and expected > 0:
                                                count_ok = all(c == expected for c in uid_counts.values())
                                                if b_micro % expected == 0:
                                                    prompt_count_expected = b_micro // expected
                                                    run_size_ok = all(run_len == expected for _, run_len in run_lengths)
                                                    run_count_ok = len(run_lengths) == prompt_count_expected
                                                    run_uid_unique_ok = len({k for k, _ in run_lengths}) == len(run_lengths)
                                                    order_ok = bool(run_size_ok and run_count_ok and run_uid_unique_ok)
                                                else:
                                                    count_ok = False
                                                    order_ok = False

                                            if count_ok is False:
                                                uid_ok = False
                                            if order_ok is False:
                                                uid_ok = False

                                            preview_n = min(16, len(uid_list))
                                            run_preview_n = min(8, len(run_lengths))
                                            print(
                                                "[composer-debug][uid] micro_batch check: "
                                                f"mini_idx={batch_idx} micro_idx={micro_debug_idx} "
                                                f"B_micro={b_micro} rollout_n={expected} "
                                                f"rollout_n_src={expected_source} "
                                                f"unique_uids={len(uid_counts)} "
                                                f"expected_prompts={prompt_count_expected} "
                                                f"count_ok={count_ok} order_ok={order_ok} "
                                                f"count_hist={dict(sorted(count_hist.items()))} "
                                                f"uid[:{preview_n}]={uid_list[:preview_n]} "
                                                f"runs[:{run_preview_n}]={run_lengths[:run_preview_n]}"
                                            )

                                    if strict_uid and not uid_ok:
                                        raise ValueError(
                                            "UID microbatch grouping check failed. "
                                            "Set GRPO_COMPOSER_DEBUG_UID_STRICT=0 to log-only mode."
                                        )

                            entropy_coeff = float(getattr(self.config, "entropy_coeff", 0.0))
                            loss_agg_mode = getattr(self.config, "loss_agg_mode", "token-mean")
                            calculate_entropy_cfg = bool(getattr(self.config, "calculate_entropy", False))
                            calculate_entropy = calculate_entropy_cfg or (entropy_coeff != 0)

                            if self.config.use_dynamic_bsz:
                                loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            else:
                                loss_scale_factor = 1 / self.gradient_accumulation

                            forward_output = self._forward_micro_batch(
                                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                            )
                            if not isinstance(forward_output, (tuple, list)):
                                raise TypeError("Expected tuple output from _forward_micro_batch.")
                            if len(forward_output) == 3:
                                entropy, log_prob, forward_hidden_states = forward_output
                            elif len(forward_output) == 2:
                                entropy, log_prob = forward_output
                                forward_hidden_states = None
                            else:
                                raise ValueError(
                                    "Unexpected _forward_micro_batch output length: "
                                    f"{len(forward_output)}"
                                )

                            if hidden_states is None:
                                hidden_states = forward_hidden_states
                            if hidden_states is None:
                                raise ValueError(
                                    "Couldn't find hidden states in micro-batch inputs or forward output."
                                )

                            if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                                old_log_prob = model_inputs["old_log_probs"]
                            else:
                                if on_policy:
                                    old_log_prob = log_prob.detach()
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]

                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                            loss_extra_kwargs: dict[str, Any] = {}
                            token_level_rewards = model_inputs.get("token_level_rewards", None)
                            uid = model_inputs.get("uid", None)
                            seq_rewards = None
                            if token_level_rewards is not None:
                                loss_extra_kwargs["token_level_rewards"] = token_level_rewards
                                loss_extra_kwargs["composer_token_level_rewards"] = token_level_rewards
                                if (
                                    isinstance(token_level_rewards, torch.Tensor)
                                    and isinstance(response_mask, torch.Tensor)
                                    and token_level_rewards.shape == response_mask.shape
                                ):
                                    seq_rewards = (token_level_rewards * response_mask).sum(dim=-1)
                                    loss_extra_kwargs["sequence_rewards"] = seq_rewards
                                    loss_extra_kwargs["composer_sequence_rewards"] = seq_rewards
                            if uid is not None:
                                loss_extra_kwargs["uid"] = uid
                                loss_extra_kwargs["composer_uid"] = uid

                            ref_log_prob = model_inputs.get("ref_log_prob", None)
                            if ref_log_prob is not None:
                                loss_extra_kwargs["ref_log_prob"] = ref_log_prob
                                loss_extra_kwargs["composer_ref_log_probs"] = ref_log_prob

                            if daro_enabled:
                                mu_id_row = model_inputs.get("daro_mu_id_row", None)
                                inv_group_tokens_row = model_inputs.get("daro_inv_group_tokens_row", None)
                                active_mu_ids = mini_active_mu_ids_tensor

                                if not isinstance(mu_id_row, torch.Tensor):
                                    raise ValueError(
                                        "DARO requires microbatch daro_mu_id_row tensor from mini-batch context."
                                    )
                                if not isinstance(inv_group_tokens_row, torch.Tensor):
                                    raise ValueError(
                                        "DARO requires microbatch daro_inv_group_tokens_row tensor from mini-batch context."
                                    )
                                if mu_id_row.ndim != 1 or mu_id_row.shape[0] != int(response_mask.shape[0]):
                                    raise ValueError(
                                        "DARO microbatch daro_mu_id_row shape mismatch: "
                                        f"{tuple(mu_id_row.shape)} vs B_micro={int(response_mask.shape[0])}"
                                    )
                                if (
                                    inv_group_tokens_row.ndim != 1
                                    or inv_group_tokens_row.shape[0] != int(response_mask.shape[0])
                                ):
                                    raise ValueError(
                                        "DARO microbatch daro_inv_group_tokens_row shape mismatch: "
                                        f"{tuple(inv_group_tokens_row.shape)} vs B_micro={int(response_mask.shape[0])}"
                                    )
                                if active_mu_ids is None:
                                    active_mu_ids = torch.empty(
                                        (0,),
                                        device=response_mask.device,
                                        dtype=torch.long,
                                    )
                                else:
                                    active_mu_ids = active_mu_ids.to(
                                        device=response_mask.device,
                                        dtype=torch.long,
                                    )
                                mu_id_row = mu_id_row.to(
                                    device=response_mask.device,
                                    dtype=torch.long,
                                )
                                inv_group_tokens_row = inv_group_tokens_row.to(
                                    device=response_mask.device,
                                    dtype=token_level_rewards.dtype,
                                )

                                loss_extra_kwargs["daro_mu_id_row"] = mu_id_row
                                loss_extra_kwargs["composer_daro_mu_id_row"] = mu_id_row
                                loss_extra_kwargs["daro_inv_group_tokens_row"] = inv_group_tokens_row
                                loss_extra_kwargs["composer_daro_inv_group_tokens_row"] = inv_group_tokens_row
                                loss_extra_kwargs["daro_active_mu_ids"] = active_mu_ids
                                loss_extra_kwargs["composer_daro_active_mu_ids"] = active_mu_ids
                                if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                                    preview_n = min(8, int(mu_id_row.shape[0]))
                                    print(
                                        "[composer-debug][daro] micro_batch payload: "
                                        f"micro_idx={micro_debug_idx} "
                                        f"B_micro={int(response_mask.shape[0])} "
                                        f"active_mu_ids={active_mu_ids.detach().cpu().tolist()} "
                                        f"mu_id_row[:{preview_n}]={mu_id_row[:preview_n].detach().cpu().tolist()} "
                                        f"inv_N_row[:{preview_n}]={inv_group_tokens_row[:preview_n].detach().cpu().tolist()}"
                                    )

                            if rollout_n is not None:
                                loss_extra_kwargs["n"] = rollout_n
                                loss_extra_kwargs["rollout_n"] = rollout_n

                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                hidden_states=hidden_states,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                                **loss_extra_kwargs,
                            )
                            pg_metrics.pop("actor/pg_loss", None)
                            pg_metrics.pop("actor/kl_loss", None)
                            micro_batch_metrics.update(pg_metrics)

                            rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                            if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                                rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                    log_prob=log_prob,
                                    rollout_log_prob=rollout_log_prob,
                                    response_mask=response_mask,
                                )
                                micro_batch_metrics.update(rollout_corr_metrics)

                            policy_loss = pg_loss
                            if calculate_entropy and entropy is not None:
                                entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                                micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                                if entropy_coeff != 0:
                                    policy_loss -= entropy_agg * entropy_coeff

                            if self.config.use_kl_loss:
                                ref_log_prob = model_inputs["ref_log_prob"]
                                kld = kl_penalty(
                                    logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                                )
                                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                            loss = policy_loss * loss_scale_factor
                            if self.scaler is not None:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                            metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                            append_to_dict(metrics, micro_batch_metrics)
                            micro_debug_idx += 1

                        grad_norm = self._optimizer_step()
                        mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                        append_to_dict(metrics, mini_batch_metrics)

                self.actor_optimizer.zero_grad()
                return metrics
            finally:
                if clear_batch_context is not None:
                    try:
                        clear_batch_context()
                    except Exception:
                        pass

        # ------------------------------------------------------------------
        # _forward_micro_batch
        # ------------------------------------------------------------------
        def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):  # type: ignore[override]
            response_length = micro_batch["responses"].size(-1)

            multi_modal_inputs = {}
            if "multi_modal_inputs" in micro_batch:
                from verl.utils.model import extract_multi_modal_inputs
                multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                input_ids = micro_batch["input_ids"]
                batch_size, seqlen = input_ids.shape
                attention_mask = micro_batch["attention_mask"]
                position_ids = micro_batch["position_ids"]

                entropy = None

                if position_ids.dim() == 3:
                    position_ids = position_ids.transpose(0, 1)

                # =========================
                # RMPAD PATH
                # =========================
                if self.use_remove_padding:

                    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask
                    )
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                    if position_ids.dim() == 3:
                        position_ids_rmpad = (
                            index_first_axis(rearrange(position_ids, "c b s -> (b s) c"), indices)
                            .transpose(0, 1)
                            .unsqueeze(1)
                        )
                    else:
                        position_ids_rmpad = index_first_axis(
                            rearrange(position_ids.unsqueeze(-1), "b s -> (b s)"), indices
                        ).transpose(0, 1)

                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                    if self.use_ulysses_sp:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad_rolled,
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                    extra_args = {}
                    if self.use_fused_kernels:
                        extra_args["temperature"] = temperature
                        extra_args["return_dict"] = True

                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                        **extra_args,
                    )

                    # ===== log probs =====
                    if self.use_fused_kernels:
                        log_probs = output.log_probs.squeeze(0)
                        entropy_rmpad = output.entropy.squeeze(0) if calculate_entropy else None

                        if not hasattr(output, "hidden_states") or output.hidden_states is None:
                            raise ValueError("Hidden states not available with fused kernels")
                        hidden_rmpad = output.hidden_states[-1]

                    else:
                        logits_rmpad = output.logits.squeeze(0)
                        logits_rmpad.div_(temperature)

                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=not calculate_entropy,
                        )

                        if calculate_entropy:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                        if not hasattr(output, "hidden_states") or output.hidden_states is None:
                            raise ValueError("Hidden states not available without fused kernels")
                        hidden_rmpad = output.hidden_states[-1]  # (total_nnz, d)

                    # ===== gather sp =====
                    if self.use_ulysses_sp:
                        log_probs = gather_outputs_and_unpad(log_probs, 0, 0, pad_size)
                        hidden_rmpad = gather_outputs_and_unpad(hidden_rmpad, 0, 0, pad_size)
                        if calculate_entropy:
                            entropy_rmpad = gather_outputs_and_unpad(entropy_rmpad, 0, 0, pad_size)

                    # ===== restore shape =====
                    full_log_probs = pad_input(
                        hidden_states=log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                    full_hidden = pad_input(
                        hidden_states=hidden_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                    if calculate_entropy:
                        full_entropy = pad_input(
                            hidden_states=entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                    # ===== slice =====
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                    hidden_states = full_hidden.squeeze(-1)[:, -response_length:, :]

                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]

                # =========================
                # NON-RMPAD PATH
                # =========================
                else:
                    extra_args = {}
                    if self.use_fused_kernels:
                        extra_args["temperature"] = temperature
                        extra_args["return_dict"] = True

                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                        **extra_args,
                    )

                    hidden_states_all = output.hidden_states[-1]  # (B, T, d)

                    if self.use_fused_kernels:
                        log_probs = output.log_probs[:, -response_length - 1 : -1]
                        entropy = output.entropy[:, -response_length - 1 : -1] if calculate_entropy else None

                    else:
                        logits = output.logits
                        logits.div_(temperature)
                        logits = logits[:, -response_length - 1 : -1, :]
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                        if calculate_entropy:
                            entropy = verl_F.entropy_from_logits(logits)

                    hidden_states = hidden_states_all[:, -response_length:, :]

            return entropy, log_probs, hidden_states

        # ------------------------------------------------------------------
        # compute_log_prob
        # ------------------------------------------------------------------
        @GPUMemoryLogger(role="dp actor", logger=logger)
        def compute_log_prob(self, data, calculate_entropy=False):  # type: ignore[override]
            self.actor_module.eval()

            micro_batch_size = data.meta_info["micro_batch_size"]
            temperature = data.meta_info["temperature"]
            use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

            data = data.select(
                batch_keys=["responses", "input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=non_tensor_select_keys,
            )

            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                micro_batches = data.split(micro_batch_size)

            log_probs_lst = []
            entropy_lst = []
            hidden_lst = []

            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                with torch.no_grad():
                    entropy, log_probs, hidden_states = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                log_probs_lst.append(log_probs)
                hidden_lst.append(hidden_states)

                if calculate_entropy:
                    entropy_lst.append(entropy)

            log_probs = torch.concat(log_probs_lst, dim=0)
            hidden_states = torch.concat(hidden_lst, dim=0)

            entropys = None
            if calculate_entropy:
                entropys = torch.concat(entropy_lst, dim=0)

            if use_dynamic_bsz:
                log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
                hidden_states = restore_dynamic_batch(hidden_states, batch_idx_list)
                if calculate_entropy:
                    entropys = restore_dynamic_batch(entropys, batch_idx_list)

            # expose embeddings
            data.batch["hidden_states"] = hidden_states
            data.batch["response_hidden_states"] = hidden_states

            return log_probs, entropys, hidden_states


__all__ = ["ComposerDataParallelPPOActor"]
