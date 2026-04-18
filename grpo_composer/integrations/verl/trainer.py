"""Custom VERL trainer extensions for grpo_composer.

This module adds a single extensible trainer surface on top of VERL's
RayPPOTrainer and patches module-level compute_advantage to support
composer-specific advantage/reward contexts.
"""

from __future__ import annotations

from abc import ABC
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import time
from tqdm import tqdm
import os
import json
import numpy as np
import torch
from .reward_ranker import BaseRanker, HeuristicRanker, RRMRanker, ensure_reward_ranks
from .patch_dp_actor import _patch_dp_actor_update_policy, unpatch_dp_actor_update_policy, _patch_dp_actor_forward_microbatch_compute_log_prob
from .rewards_registery import _REWARD_TRANSFORMS, _sequence_rewards_from_token
from .patch_fsdp_workers import _patch_fsdp_compute_log_probs
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

_VERL_IMPORT_ERROR: Optional[Exception] = None
try:
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, apply_kl_penalty
except Exception as exc:  # pragma: no cover - exercised when verl is absent
    _VERL_IMPORT_ERROR = exc
    core_algos = None
    AdvantageEstimator = None
    ray_trainer_module = None

    class RayPPOTrainer:  # type: ignore[override]
        """Fallback stub so this module can be imported without verl installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ComposerRayPPOTrainer requires `verl` to be installed. "
                f"Original import error: {_VERL_IMPORT_ERROR!r}"
            )


_ORIGINAL_COMPUTE_ADVANTAGE = None
_ORIGINAL_RAY_TRAINER_CLASS = None
_ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = None


def _is_wandb_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    return (
        "401" in msg
        or "not logged in" in msg
        or "permission_error" in msg
        or "upsertbucket" in msg
        or exc_name == "commerror"
    )


def _to_metric_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            try:
                return float(value.detach().item())
            except Exception:
                return None
        return None
    return None


class _StepMetricsCsvWriter:
    """Append step metrics in long CSV format: wall_time,step,metric,value."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = self.path.with_name(f"{self.path.stem}_{ts}{self.path.suffix}")
            self.path.rename(backup)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["wall_time", "step", "metric", "value"])
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except Exception:
            pass

    def write(self, step: Any, metrics: Any) -> None:
        if not isinstance(metrics, dict):
            return
        wall_time = time.time()
        wrote = False
        
        # Also print core metrics to console so user can see them immediately
        # even if Ray Actor is SIGKILLed before file closes.
        if step is not None:
            core_metrics = {
                k: v for k, v in metrics.items() 
                if k in ("actor/entropy", "actor/clip_frac", "actor/approx_kl_with_ref", "critic/loss", "reward/composer_sequence_rewards_mean") or k.startswith("training/")
            }
            core_str = " | ".join(f"{k}: {_to_metric_scalar(v):.4f}" for k, v in core_metrics.items() if _to_metric_scalar(v) is not None)
            if core_str:
                print(f"[{time.strftime('%H:%M:%S')}] Step {step} Metrics: {core_str}")

        for metric_name, metric_value in metrics.items():
            scalar = _to_metric_scalar(metric_value)
            if scalar is None:
                continue
            self._writer.writerow([wall_time, step, str(metric_name), scalar])
            wrote = True
        
        if wrote:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno()) # Force network sync for Modal Volumes!
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:
            pass


def _build_tracking_with_csv_fallback(original_tracking_cls: Any, csv_path: str):
    class _TrackingWithCsvFallback:
        def __init__(self, *args, **kwargs):
            self._csv = _StepMetricsCsvWriter(csv_path)
            self._impl = None
            if original_tracking_cls is None:
                print(
                    "[grpo_composer] Tracking backend: csv-only "
                    f"(no upstream Tracking class). csv={csv_path}"
                )
                return
            try:
                # veRL's Tracking.__init__ calls wandb.init() without
                # wandb.login() first. In Ray worker processes the .netrc
                # file doesn't exist, so wandb.init() fails with a 401
                # upsertBucket error even when WANDB_API_KEY is in the env.
                # Explicitly login before delegating to the Tracking class.
                _wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
                if _wandb_key:
                    try:
                        import wandb
                        wandb.login(key=_wandb_key, relogin=True)
                    except Exception as _login_exc:
                        print(f"[grpo_composer] wandb.login() failed: {_login_exc}")
                self._impl = original_tracking_cls(*args, **kwargs)
                print(
                    "[grpo_composer] Tracking backend: upstream+csv "
                    f"(W&B expected if configured). csv={csv_path}"
                )
            except Exception as exc:
                reason = "auth/permission (INVALID KEY OR ENTITY)" if _is_wandb_auth_error(exc) else "tracking backend"
                if _is_wandb_auth_error(exc):
                    print(
                        "\n🚨 [CRITICAL W&B ERROR] WANDB_API_KEY is present in the environment, "
                        "but the W&B backend rejected it with HTTP 401 'user is not logged in'.\n"
                        "This means exactly one of two things:\n"
                        "  1. The API key in your Modal Secret is expired, revoked, or has leading/trailing spaces.\n"
                        "  2. You are setting WANDB_ENTITY to a team name you do not have permission to write to.\n"
                        "Please go to your Modal Secrets dashboard, delete the 'wandb' secret, and re-create it perfectly!\n"
                    )
                print(
                    "[grpo_composer] Tracking init failed "
                    f"({reason}). Continuing with CSV-only logging at {csv_path}. error={exc}"
                )

        def log(self, data, step=None, *args, **kwargs):
            self._csv.write(step, data)
            if self._impl is not None:
                try:
                    return self._impl.log(data, step=step, *args, **kwargs)
                except Exception as exc:
                    reason = "auth/permission" if _is_wandb_auth_error(exc) else "tracking backend"
                    print(
                        "[grpo_composer] Tracking log failed "
                        f"({reason}). Continuing CSV-only. error={exc}"
                    )
                    self._impl = None
                    return None
            return None

        def finish(self):
            try:
                if self._impl is not None and hasattr(self._impl, "finish"):
                    self._impl.finish()
            finally:
                self._csv.close()

    return _TrackingWithCsvFallback


def _strict_validation_enabled() -> bool:
    return os.environ.get("GRPO_COMPOSER_STRICT_VALIDATION", "1") != "0"


def _shape_debug(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"torch{tuple(value.shape)}"
    if isinstance(value, np.ndarray):
        return f"np{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def _cfg_get(config: Any, key: str, default=None):
    from .loss_context import get_composer_config

    val = None
    if config is not None:
        getter = getattr(config, "get", None)
        if callable(getter):
            try:
                val = getter(key, None)
            except TypeError:
                pass
        if val is None:
            val = getattr(config, key, None)

    if val is not None:
        return val

    # Fallback to globally injected composer config
    composer_cfg = get_composer_config()
    if key in composer_cfg and composer_cfg[key] is not None:
        return composer_cfg[key]

    return default


def _cfg_get_nested(config: Any, path: tuple[str, ...], default=None):
    current = config
    for part in path:
        if current is None:
            return default
        current = _cfg_get(current, part, None)
    return default if current is None else current


def _to_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _dapo_debug_enabled(config: Any) -> bool:
    cfg_flag = _cfg_get_nested(config, ("algorithm", "filter_groups", "debug"), None)
    if cfg_flag is not None:
        return _to_bool_flag(cfg_flag, default=False)
    return _to_bool_flag(os.environ.get("GRPO_COMPOSER_DAPO_DEBUG", "0"), default=False)


def _dapo_debug(config: Any, message: str) -> None:
    if _dapo_debug_enabled(config):
        print(f"[DAPO DEBUG] {message}", flush=True)


def _maybe_get(data: Any, key: str):
    if hasattr(data, "batch") and key in data.batch.keys():
        return data.batch[key]
    if hasattr(data, "non_tensor_batch") and key in data.non_tensor_batch:
        return data.non_tensor_batch[key]
    return None


def _set_batch_tensor(data: Any, key: str, value: torch.Tensor) -> None:
    if not hasattr(data, "batch"):
        raise ValueError("Data object missing `batch` for tensor assignment")
    data.batch[key] = value


def _set_non_tensor(data: Any, key: str, value: Any) -> None:
    if not hasattr(data, "non_tensor_batch"):
        raise ValueError("Data object missing `non_tensor_batch` for non-tensor assignment")
    data.non_tensor_batch[key] = value


def _get_uid_groups(data: Any, batch_size: int) -> dict[Any, list[int]]:
    uid = None
    if hasattr(data, "non_tensor_batch") and "uid" in data.non_tensor_batch:
        uid = data.non_tensor_batch["uid"]
    elif hasattr(data, "batch") and "uid" in data.batch.keys():
        uid = data.batch["uid"]

    if uid is None:
        return {i: [i] for i in range(batch_size)}

    if isinstance(uid, torch.Tensor):
        uid_arr = uid.detach().cpu().numpy()
    else:
        uid_arr = np.asarray(uid)

    if uid_arr.ndim != 1 or uid_arr.shape[0] != batch_size:
        raise ValueError(f"uid must be shape (bs,), got {uid_arr.shape} for batch size {batch_size}")

    groups: dict[Any, list[int]] = defaultdict(list)
    for i, key in enumerate(uid_arr.tolist()):
        groups[key].append(i)
    return groups




def _parse_reward_pipeline(config: Any) -> list[str]:
    pipeline = _cfg_get(config, "composer_reward_pipeline", None)
    if pipeline is None:
        pipeline = _cfg_get(config, "reward_pipeline", None)

    if pipeline is None:
        nested = _cfg_get(config, "composer", None)
        pipeline = _cfg_get(nested, "reward_pipeline", [])

    if isinstance(pipeline, str):
        pipeline = [item.strip() for item in pipeline.split(",") if item.strip()]
    return list(pipeline or [])


def _reward_pipeline_requires_hidden_states(config: Any) -> bool:
    return "diversity_adjusted" in set(_parse_reward_pipeline(config))


def _apply_reward_pipeline(data: Any, config: Any) -> Any:
    pipeline = _parse_reward_pipeline(config)
    
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"🛠️ [DEBUG] Loading Reward Pipeline: {pipeline}")
        
    for transform_name in pipeline:
        transform = _REWARD_TRANSFORMS.get(transform_name)
        if transform is None:
            raise ValueError(
                f"Unknown composer reward transform '{transform_name}'. "
                f"Available: {sorted(_REWARD_TRANSFORMS.keys())}"
            )
            
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(f"🛠️ [DEBUG] Executing Reward Transform: {transform_name}")
            
        transform(data, config)
    return data


def _inject_standard_composer_context(data: Any) -> None:
    token_level_rewards = _maybe_get(data, "token_level_rewards")
    response_mask = _maybe_get(data, "response_mask")
    if isinstance(token_level_rewards, torch.Tensor) and isinstance(response_mask, torch.Tensor):
        if token_level_rewards.shape == response_mask.shape:
            sequence_rewards = _sequence_rewards_from_token(token_level_rewards, response_mask)
            _set_batch_tensor(data, "composer_sequence_rewards", sequence_rewards)

    # Carry commonly used auxiliary tensors under composer_* aliases.
    for src_key, dst_key in [
        ("old_log_probs", "composer_old_log_probs"),
        ("reward_baselines", "composer_reward_baselines"),
        ("sum_pi_squared", "composer_sum_pi_squared"),
        ("reference_rewards", "composer_reference_rewards"),
        ("multi_rewards", "composer_multi_rewards"),
        ("strata", "composer_strata"),
        ("stratum_ids", "composer_strata"),
        ("log_probs_aug", "composer_log_probs_aug"),
        ("mask_aug", "composer_mask_aug"),
    ]:
        value = _maybe_get(data, src_key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            _set_batch_tensor(data, dst_key, value)
        else:
            _set_non_tensor(data, dst_key, value)


def _collect_adv_optional_kwargs(data: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if hasattr(data, "non_tensor_batch") and "uid" in data.non_tensor_batch:
        kwargs["index"] = data.non_tensor_batch["uid"]

    for batch_key, kwarg_key in [
        ("reward_baselines", "reward_baselines"),
        ("sum_pi_squared", "sum_pi_squared"),
        ("rollout_is_weights", "rollout_is_weights"),
        ("composer_multi_rewards", "multi_rewards"),
        ("multi_rewards", "multi_rewards"),
        ("composer_reference_rewards", "reference_rewards"),
        ("reference_rewards", "reference_rewards"),
        ("composer_old_log_probs", "old_log_probs"),
        ("old_log_probs", "old_log_probs"),
        ("composer_strata", "strata"),
        ("strata", "strata"),
        ("stratum_ids", "strata"),
    ]:
        value = _maybe_get(data, batch_key)
        
        # If veRL stripped it from `.batch`, check `.non_tensor_batch` (used for PVPO reference hook)
        if value is None and hasattr(data, "non_tensor_batch") and batch_key in data.non_tensor_batch:
            value = data.non_tensor_batch[batch_key]
            
        if value is not None:
            kwargs[kwarg_key] = value

    return kwargs

def build_ranker(composer_cfg: Any) -> BaseRanker | None:
    ranking_type = str(_cfg_get(composer_cfg, "ranking_type", "")).strip().lower()
    if not ranking_type:
        return None
    if ranking_type == "heuristic":
        return HeuristicRanker()
    if ranking_type == "rrm":
        worker = _cfg_get(composer_cfg, "ranking_worker", None)
        if worker is None:
            raise ValueError(
                "ranking_type='rrm' requires an initialized ranking_worker (Ray actor). "
                "Use ranking_type='heuristic' for the current safe integration."
            )
        return RRMRanker(worker)
    raise ValueError(f"Unknown ranking_type '{ranking_type}'. Expected one of: heuristic, rrm")


def composer_compute_advantage(
    data: Any,
    adv_estimator: Any,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[Any] = None,
    ranker: BaseRanker | None = None,
    tokenizer: Any = None,
):
    if core_algos is None or AdvantageEstimator is None or ray_trainer_module is None:
        raise RuntimeError(
            "composer_compute_advantage requires verl to be installed. "
            f"Original import error: {_VERL_IMPORT_ERROR!r}"
        )

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print("🛠️ [DEBUG] composer_compute_advantage IS BEING CALLED SUCCESSFULLY!")

    # Ensure custom advantage estimators are registered in this process
    # (TaskRunner is a separate Ray actor where train_grpo.py's top-level
    # import hasn't run).
    import importlib

    importlib.import_module("grpo_composer.integrations.verl.advantages")

    # ── uid fixup ──────────────────────────────────────────────────────
    # veRL's RayPPOTrainer.fit() assigns uid *before* DataProto.repeat(),
    # expecting repeat(interleave=True) to replicate non_tensor_batch.
    # Some veRL versions do NOT replicate non_tensor_batch, leaving every
    # row with a unique uid (hist={1: total_rows}) — no grouping at all.
    # Detect this and reconstruct per-prompt uids from num_repeat.
    resolved_num_repeat = int(num_repeat) if num_repeat is not None else 1
    if resolved_num_repeat <= 1:
        meta_info = getattr(data, "meta_info", None)
        for _mi_key in ("rollout_n", "n"):
            _candidate = None
            if isinstance(meta_info, dict):
                _candidate = meta_info.get(_mi_key)
            else:
                _getter = getattr(meta_info, "get", None)
                if callable(_getter):
                    try:
                        _candidate = _getter(_mi_key, None)
                    except Exception:
                        pass
            if _candidate is not None:
                try:
                    _candidate_int = int(_candidate)
                except Exception:
                    _candidate_int = 1
                if _candidate_int > 1:
                    resolved_num_repeat = _candidate_int
                    break

    if resolved_num_repeat <= 1:
        _cfg_rollout_n = _cfg_get_nested(config, ("actor_rollout_ref", "rollout", "n"), None)
        if _cfg_rollout_n is not None:
            try:
                _cfg_rollout_n = int(_cfg_rollout_n)
            except Exception:
                _cfg_rollout_n = 1
            if _cfg_rollout_n > 1:
                resolved_num_repeat = _cfg_rollout_n

    if resolved_num_repeat <= 1:
        _composer_rollout_n = _cfg_get(_cfg_get(config, "composer", None), "rollout_n", None)
        if _composer_rollout_n is not None:
            try:
                _composer_rollout_n = int(_composer_rollout_n)
            except Exception:
                _composer_rollout_n = 1
            if _composer_rollout_n > 1:
                resolved_num_repeat = _composer_rollout_n

    if resolved_num_repeat <= 1:
        _cfg_env = os.environ.get("GRPO_COMPOSER_ROLLOUT_N")
        if _cfg_env is not None:
            try:
                _env_n = int(_cfg_env)
            except ValueError:
                _env_n = 1
            if _env_n > 1:
                resolved_num_repeat = _env_n

    _has_uid = (
        hasattr(data, "non_tensor_batch")
        and data.non_tensor_batch is not None
        and "uid" in data.non_tensor_batch
    )
    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(f"[composer-debug] uid fixup check: num_repeat={resolved_num_repeat} has_uid={_has_uid}")
    if resolved_num_repeat > 1 and _has_uid:
        uid_raw = data.non_tensor_batch["uid"]
        uid_arr = np.asarray(uid_raw)
        total_rows = uid_arr.shape[0]
        unique_count = len(set(uid_arr.tolist()))
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(f"[composer-debug] uid fixup pre: total={total_rows} unique={unique_count}")
        if unique_count == total_rows and total_rows % resolved_num_repeat == 0:
            # Every row is unique → repeat() didn't propagate non_tensor_batch.
            # Reconstruct: rows are interleaved, so row i belongs to prompt i // num_repeat.
            num_prompts = total_rows // resolved_num_repeat
            fixed_uid = np.array(
                [uid_arr[i * resolved_num_repeat] for i in range(num_prompts) for _ in range(resolved_num_repeat)],
                dtype=uid_arr.dtype,
            )
            data.non_tensor_batch["uid"] = fixed_uid
            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                fixed_unique = len(set(fixed_uid.tolist()))
                print(
                    f"[composer-debug] uid fixup: {total_rows} rows had {unique_count} unique uids "
                    f"→ reconstructed {fixed_unique} unique uids (num_repeat={resolved_num_repeat})"
                )

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        print(data.batch.keys())

    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = ray_trainer_module.compute_response_mask(data)

    needs_hidden_states = _reward_pipeline_requires_hidden_states(config)
    has_hidden_states = "hidden_states" in data.batch and data.batch["hidden_states"] is not None
    if needs_hidden_states and not has_hidden_states:
        raise ValueError(
            "Hidden states missing before advantage computation for diversity_adjusted reward pipeline."
        )

    if has_hidden_states and "response_hidden_states" not in data.batch:
        data.batch["response_hidden_states"] = data.batch["hidden_states"]
        
    if ranker is not None:
        ensure_reward_ranks(
            data,
            ranker,
            tokenizer=tokenizer,
            config=config,
            debug=os.environ.get("GRPO_COMPOSER_DEBUG") == "1",
        )
    
    data = _apply_reward_pipeline(data, config)
    _inject_standard_composer_context(data)

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if _cfg_get(config, "use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                _cfg_get(_cfg_get(config, "pf_ppo", None), "reweight_method", None),
                _cfg_get(_cfg_get(config, "pf_ppo", None), "weight_pow", None),
            )
        return data

    if adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs: dict[str, Any] = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index" : data.non_tensor_batch["uid"],
        "config": config,
    }
    adv_kwargs.update(_collect_adv_optional_kwargs(data))
    
    # Optional explicitly passed variables for custom estimators like PVPO
    if "reference_rewards" in data.batch:
        adv_kwargs["reference_rewards"] = data.batch["reference_rewards"]
    elif "composer_reference_rewards" in data.batch:
        adv_kwargs["composer_reference_rewards"] = data.batch["composer_reference_rewards"]

    advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@dataclass
class FlowRuntimeContext:
    """Mutable runtime context shared across flow plugins."""

    metrics: dict[str, Any] = field(default_factory=dict)


class FlowPlugin(ABC):
    """Minimal flow plugin protocol used by the trainer extension points.

    Three hooks cover every runtime extension we currently need:

    * ``configure``: one-shot setup at trainer init (capture tokenizer,
      validate prerequisites, etc.).
    * ``before_generate``: mutate / replace the rollout batch immediately
      before ``generate_sequences`` (e.g. Info-GRPO latent-seed injection).
    * ``before_compute_advantage``: mutate the post-rollout DataProto before
      advantages are computed (e.g. PVPO/GAPO reference-reward generation).
    """

    def configure(self, trainer: "ComposerRayPPOTrainer") -> None:
        return None

    def before_generate(self, trainer: "ComposerRayPPOTrainer", batch: Any) -> Any:
        return batch

    def before_compute_advantage(self, trainer: "ComposerRayPPOTrainer", data: Any) -> Any:
        return data


class PassThroughFlowPlugin(FlowPlugin):
    """No-op plugin used when a flow only affects runtime hooks in `fit()`."""


_FLOW_PLUGIN_REGISTRY: dict[str, type[FlowPlugin]] = {
    "default": PassThroughFlowPlugin,
    "pvpo": PassThroughFlowPlugin,
    "pvpo_grpo": PassThroughFlowPlugin,
    "gapo": PassThroughFlowPlugin,
    "gapo_grpo": PassThroughFlowPlugin,
    "info_grpo": PassThroughFlowPlugin,
}

def _parse_flow_list(config: Any) -> list[str]:
    algorithm = _cfg_get(config, "algorithm", None)
    composer = _cfg_get(config, "composer", None)
    flow = _cfg_get(algorithm, "composer_flow", None)
    if flow is None:
        flow = _cfg_get(algorithm, "flow", None)
    if flow is None:
        flow = _cfg_get(composer, "composer_flow", None)
    if flow is None:
        flow = _cfg_get(composer, "flow", None)

    plugin_names = _cfg_get(algorithm, "composer_flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(algorithm, "flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(composer, "composer_flow_plugins", None)
    if plugin_names is None:
        plugin_names = _cfg_get(composer, "flow_plugins", None)

    parsed_plugins: list[str] = []
    if isinstance(plugin_names, str):
        parsed_plugins = [name.strip() for name in plugin_names.split(",") if name.strip()]
    elif plugin_names:
        parsed_plugins = [str(name) for name in plugin_names]

    if flow is None:
        flow = "default"
    flow = str(flow)
    if flow and flow not in parsed_plugins:
        parsed_plugins.insert(0, flow)

    if not parsed_plugins:
        parsed_plugins = ["default"]
    return parsed_plugins



# Ensure worker-side config binding when this module is imported via
# actor_rollout_ref.model.external_lib in FSDP worker processes.
_patch_dp_actor_update_policy()
_patch_dp_actor_forward_microbatch_compute_log_prob()
_patch_fsdp_compute_log_probs()

class ComposerRayPPOTrainer(RayPPOTrainer):
    """Single custom trainer that extends VERL's RayPPOTrainer with flow plugins."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inject_composer_config()
        self.composer_context = FlowRuntimeContext()
        self.composer_flow_plugins = self._build_flow_plugins()
        for plugin in self.composer_flow_plugins:
            plugin.configure(self)
        self._validate_supported_modes()

    def _inject_composer_config(self) -> None:
        """Store ``composer:`` YAML keys in a module-level global.

        veRL validates ``actor_rollout_ref.actor`` as ``FSDPActorConfig`` at
        startup, rejecting unknown keys.  We cannot merge composer-specific
        keys (clip_mode, agg_mode, regularizer, …) into the actor OmegaConf
        tree — doing so would break the dataclass conversion in FSDP workers.

        Instead we store them in the shared loss context, which the loss
        function reads via ``_config_get(config, key, default)``'s fallback.
        """
        composer_cfg = _cfg_get(self.config, "composer", None)
        if composer_cfg is None:
            return

        from .loss_context import set_composer_config

        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(composer_cfg):
                config_dict = OmegaConf.to_container(composer_cfg, resolve=True)
            else:
                config_dict = dict(composer_cfg) if not isinstance(composer_cfg, dict) else composer_cfg
        except ImportError:
            config_dict = dict(composer_cfg) if not isinstance(composer_cfg, dict) else composer_cfg

        set_composer_config(config_dict)

        # Also store the raw dict so we can explicitly pass it via the DataBatch
        # metadata. Ray workers do not reliably inherit environment variables after
        # initialization, so we push it physically with the data.
        self.composer_config_dict = config_dict
        try:
            os.environ["GRPO_COMPOSER_CONFIG"] = json.dumps(config_dict)
        except Exception:
            pass

    def _build_flow_plugins(self) -> list[FlowPlugin]:
        flow_names = _parse_flow_list(self.config)
        plugins: list[FlowPlugin] = []
        for flow_name in flow_names:
            cls = _FLOW_PLUGIN_REGISTRY.get(flow_name)
            if cls is None:
                raise ValueError(
                    f"Unknown composer flow plugin '{flow_name}'. "
                    f"Available: {sorted(_FLOW_PLUGIN_REGISTRY.keys())}"
                )
            plugins.append(cls())
        return plugins

    def _validate_supported_modes(self) -> None:
        self._enforce_balance_batch_disabled()

    def _enforce_balance_batch_disabled(self) -> None:
        # veRL's balance_batch reshuffles rows and breaks GRPO/PVPO grouping guarantees.
        if hasattr(self.config, "trainer") and getattr(self.config.trainer, "balance_batch", False):
            import logging
            logging.getLogger(__name__).warning(
                "[grpo_composer] FORCIBLY DISABLING trainer.balance_batch. "
                "Shuffling breaks grouped rollouts and can invalidate PVPO/GRPO advantages."
            )
            self.config.trainer.balance_batch = False

    def _validate_actor_batch_contract(self, batch: Any) -> None:
        """Fail-fast validation before veRL microbatch update begins."""
        errors: list[str] = []

        response_mask = _maybe_get(batch, "response_mask")
        if not isinstance(response_mask, torch.Tensor):
            errors.append(f"`response_mask` must be torch.Tensor, got {_shape_debug(response_mask)}")
            response_shape = None
            batch_size = None
        elif response_mask.ndim != 2:
            errors.append(f"`response_mask` must be 2D [B,T], got {_shape_debug(response_mask)}")
            response_shape = tuple(response_mask.shape)
            batch_size = response_mask.shape[0]
        else:
            response_shape = tuple(response_mask.shape)
            batch_size = response_mask.shape[0]

        token_level_rewards = _maybe_get(batch, "token_level_rewards")
        if not isinstance(token_level_rewards, torch.Tensor):
            errors.append(f"`token_level_rewards` must be torch.Tensor, got {_shape_debug(token_level_rewards)}")
        elif token_level_rewards.ndim != 2:
            errors.append(f"`token_level_rewards` must be 2D [B,T], got {_shape_debug(token_level_rewards)}")
        elif response_shape is not None and tuple(token_level_rewards.shape) != response_shape:
            errors.append(
                "`token_level_rewards` must match `response_mask` shape, got "
                f"{_shape_debug(token_level_rewards)} vs {_shape_debug(response_mask)}"
            )

        advantages = _maybe_get(batch, "advantages")
        if not isinstance(advantages, torch.Tensor):
            errors.append(f"`advantages` must be torch.Tensor before actor update, got {_shape_debug(advantages)}")
        elif advantages.ndim != 2:
            errors.append(f"`advantages` must be 2D [B,T], got {_shape_debug(advantages)}")
        elif response_shape is not None and tuple(advantages.shape) != response_shape:
            errors.append(
                "`advantages` must match `response_mask` shape, got "
                f"{_shape_debug(advantages)} vs {_shape_debug(response_mask)}"
            )

        uid = _maybe_get(batch, "uid")
        if uid is not None and batch_size is not None:
            uid_len = None
            if isinstance(uid, torch.Tensor):
                if uid.ndim == 1:
                    uid_len = uid.shape[0]
            elif isinstance(uid, np.ndarray):
                if uid.ndim == 1:
                    uid_len = uid.shape[0]
            elif isinstance(uid, (list, tuple)):
                uid_len = len(uid)
            if uid_len is not None and uid_len != batch_size:
                errors.append(f"`uid` length must match batch B={batch_size}, got {uid_len}")

        composer_cfg = getattr(self, "composer_config_dict", {})
        agg_mode = _cfg_get(composer_cfg, "agg_mode", "token_mean")
        if agg_mode == "difficulty_weighted" and batch_size is not None:
            if uid is None:
                errors.append("`uid` is required for agg_mode=difficulty_weighted")

        if errors:
            raise ValueError(
                "Composer preflight validation failed before actor update:\n- "
                + "\n- ".join(errors)
            )

        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            seq_rewards = _maybe_get(batch, "composer_sequence_rewards")
            print(
                "[composer-debug] Pre-update batch contract OK: "
                f"response_mask={_shape_debug(response_mask)}, "
                f"token_level_rewards={_shape_debug(token_level_rewards)}, "
                f"advantages={_shape_debug(advantages)}, "
                f"composer_sequence_rewards={_shape_debug(seq_rewards)}"
            )

    def fit(self) -> None:
        """Override fit to guarantee compute_advantage patching within the execution loop."""
        import verl.trainer.ppo.ray_trainer as ray_trainer_module
        import verl.utils.tracking as tracking_module
        import sys
        from .info_grpo_hook import InfoGRPORolloutAugmentor
        from .ref_reward_runtime import ensure_reference_rewards, has_reference_rewards

        # Force intercepting the locally scoped compute_advantage inside ray_trainer
        original_compute_advantage = getattr(ray_trainer_module, "compute_advantage", None)
        original_compute_reward = getattr(ray_trainer_module, "compute_reward", None)
        original_tracking = getattr(ray_trainer_module, "Tracking", None)
        original_tracking_utils = getattr(tracking_module, "Tracking", None)
        tracking_base_cls = original_tracking or original_tracking_utils

        composer_cfg = getattr(self, "composer_config_dict", {})
        ref_reward_source = str(composer_cfg.get("reference_reward_source", "auto")).strip().lower()
        ranker = build_ranker(composer_cfg)
        default_local_dir = str(_cfg_get_nested(self.config, ("trainer", "default_local_dir"), "/checkpoints"))
        csv_path = str(Path(default_local_dir) / "metrics" / "training_metrics.csv")

        try:
            # Re-assert before entering veRL fit loop in case another component toggled it.
            self._enforce_balance_batch_disabled()
            print(f"[grpo_composer] Step metrics CSV path: {csv_path}")

            # Always persist scalar step metrics into CSV.
            ray_trainer_module.Tracking = _build_tracking_with_csv_fallback(tracking_base_cls, csv_path)
            if "verl.trainer.ppo.ray_trainer" in sys.modules:
                sys.modules["verl.trainer.ppo.ray_trainer"].Tracking = ray_trainer_module.Tracking
            # Patch source Tracking class too, in case veRL resolves from utils module.
            tracking_module.Tracking = ray_trainer_module.Tracking
            if "verl.utils.tracking" in sys.modules:
                sys.modules["verl.utils.tracking"].Tracking = ray_trainer_module.Tracking

            if "info_grpo" in self.composer_config_dict.get("composer_flow", ""):
                if self.async_rollout_mode:
                    original_method = self.async_rollout_manager.generate_sequences
                    self.async_rollout_manager.generate_sequences = InfoGRPORolloutAugmentor.wrap_generate_sequences(original_method, self.tokenizer)
                else:
                    original_method = self.actor_rollout_wg.generate_sequences
                    self.actor_rollout_wg.generate_sequences = InfoGRPORolloutAugmentor.wrap_generate_sequences(original_method, self.tokenizer)
            
            # Hook reference rewards if PVPO or if the config requests it dynamically
            flow_names = _parse_flow_list(self.config)
            needs_reference_reward = any(
                name in ["pvpo", "pvpo_grpo", "gapo", "gapo_grpo"] or "reference_rewards" in name 
                for name in flow_names
            )

            def hooked_compute_advantage(data, adv_estimator, *args, **kwargs):
                debug = os.environ.get("GRPO_COMPOSER_DEBUG") == "1"
                if debug:
                    print(f"[grpo_composer-debug] hooked_compute_advantage data={type(data)}")

                if needs_reference_reward and not has_reference_rewards(data):
                    ensure_reference_rewards(
                        self,
                        data,
                        composer_cfg=composer_cfg,
                        ref_reward_source=ref_reward_source,
                        debug=debug,
                    )

                return composer_compute_advantage(
                    data,
                    adv_estimator,
                    *args,
                    ranker=ranker,
                    tokenizer=self.tokenizer,
                    **kwargs,
                )

            ray_trainer_module.compute_advantage = hooked_compute_advantage
            if "verl.trainer.ppo.ray_trainer" in sys.modules:
                sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = hooked_compute_advantage

            from omegaconf import OmegaConf

            from verl.utils.tracking import Tracking

            logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

            self.global_steps = 0

            # load checkpoint before doing anything
            self._load_checkpoint()

            # perform validation before training
            # currently, we only support validation using the reward_function.
            if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                assert val_metrics, f"{val_metrics=}"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    return

            if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
                rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
                rollout_skip.wrap_generate_sequences()

            # add tqdm
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

            # we start from step 1
            self.global_steps += 1
            last_val_metrics = None
            self.max_steps_duration = 0

            prev_step_profile = False
            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            next_step_profile = False
            timing_raw = defaultdict(float)
            batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0

            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    metrics = {}

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                    rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                    prompt_bsz = int(_cfg_get_nested(self.config, ("data", "train_batch_size"), 0) or 0)
                    filter_groups_enabled = bool(_cfg_get_nested(self.config, ("algorithm", "filter_groups", "enable"), False))
                    filter_metric_name = str(_cfg_get_nested(self.config, ("algorithm", "filter_groups", "metric"), "seq_reward"))
                    max_num_gen_batches = int(
                        _cfg_get_nested(self.config, ("algorithm", "filter_groups", "max_num_gen_batches"), 0) or 0
                    )
                    reward_extra_infos_dict: dict[str, list] = {}
                    new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    num_gen_batches += 1
                    if prompt_bsz <= 0:
                        prompt_bsz = int(len(new_batch.batch))
                    _dapo_debug(
                        self.config,
                        "step="
                        + str(self.global_steps)
                        + f" gen_batch={num_gen_batches} filter_enabled={filter_groups_enabled} "
                        + f"metric={filter_metric_name} prompt_bsz={prompt_bsz} rollout_n={rollout_n}",
                    )

                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("step", timing_raw):
                        # add uid to batch
                        new_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                        )

                        gen_batch = self._get_gen_batch(new_batch)

                        # pass global_steps to trace
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                new_batch = new_batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                    rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                    new_batch = new_batch.union(rm_scores)
                                reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                new_batch.pop(batch_keys=list(keys_to_pop))

                                new_batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                        new_batch = new_batch.union(gen_batch_output)

                        if "response_mask" not in new_batch.batch.keys():
                            new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(new_batch, metrics=metrics)

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(
                                    compute_reward_async.remote(
                                        data=new_batch, config=self.config, tokenizer=self.tokenizer
                                    )
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                            new_batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                new_batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                new_batch, kl_metrics = apply_kl_penalty(
                                    new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(
                                    kl_metrics
                                )  # TODO: This will be cleared if we use multiple generation batches
                            else:
                                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        if not filter_groups_enabled:
                            batch = new_batch
                            _dapo_debug(self.config, f"filter_disabled batch_ready traj={len(batch)}")
                        else:  # NOTE: When prompts after filtering is less than train batch size,
                            # we skip to the next generation batch
                            metric_name = filter_metric_name
                            if metric_name == "seq_final_reward":
                                # Turn to numpy for easier filtering
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported algorithm.filter_groups.metric={metric_name!r}. "
                                    "Expected one of: seq_reward, seq_final_reward."
                                )
                            metric_vals = np.asarray(new_batch.non_tensor_batch[metric_name], dtype=float)
                            if metric_vals.size > 0:
                                _dapo_debug(
                                    self.config,
                                    f"metric={metric_name} traj={metric_vals.size} "
                                    + f"mean={float(metric_vals.mean()):.6f} std={float(metric_vals.std()):.6f} "
                                    + f"min={float(metric_vals.min()):.6f} max={float(metric_vals.max()):.6f}",
                                )

                            # Collect the sequence reward for each trajectory
                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            prompt_uid2metric_std = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    kept_traj_idxs.append(idx)

                            pre_filter_traj = len(new_batch)
                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                            _dapo_debug(
                                self.config,
                                f"kept_prompts={len(kept_prompt_uids)}/{len(prompt_uid2metric_vals)} "
                                + f"kept_traj={len(kept_traj_idxs)}/{pre_filter_traj} "
                                + f"accum_prompt={num_prompt_in_batch}/{prompt_bsz}",
                            )

                            if num_prompt_in_batch < prompt_bsz:
                                print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"{num_gen_batches=}. Keep generating...")
                                    _dapo_debug(
                                        self.config,
                                        f"continue_sampling num_gen_batches={num_gen_batches} "
                                        + f"max_num_gen_batches={max_num_gen_batches}",
                                    )
                                    is_last_step = self.global_steps >= self.total_training_steps
                                    continue
                                else:
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                        + " Generated too many. Please check if your data are too difficult."
                                        + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                            else:
                                # Align the batch
                                traj_bsz = prompt_bsz * rollout_n
                                batch = batch[:traj_bsz]
                                reward_extra_infos_dict = {}
                                _dapo_debug(
                                    self.config,
                                    f"batch_aligned traj_bsz={traj_bsz} actual_traj={len(batch)} "
                                    + f"num_gen_batches={num_gen_batches}",
                                )

                        if batch is None:
                            raise RuntimeError("Batch construction failed after generation/filtering.")

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                            apply_rollout_correction(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            if "token_level_scores" not in batch.batch:
                                raise ValueError(
                                    "token_level_scores missing before advantage computation."
                                )
                            if not self.config.algorithm.use_kl_in_reward and "token_level_rewards" not in batch.batch:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            output = self.actor_rollout_wg.compute_log_prob(batch)
                            hidden_states = output.batch.get("hidden_states", None)
                            if hidden_states is not None:
                                hidden_dp = DataProto.from_dict(
                                    tensors={
                                        "hidden_states": hidden_states,
                                        "response_hidden_states": hidden_states,
                                    }
                                )
                                batch = batch.union(hidden_dp)
                            elif _reward_pipeline_requires_hidden_states(self.config.algorithm):
                                raise ValueError(
                                    "Hidden states were not returned by actor_rollout_wg.compute_log_prob, "
                                    "but current reward pipeline requires them."
                                )


                            batch = hooked_compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # training metrics
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    metrics["train/num_gen_batches"] = num_gen_batches
                    _dapo_debug(
                        self.config,
                        f"step_complete step={self.global_steps} train/num_gen_batches={num_gen_batches}",
                    )
                    # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                    # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    # this is experimental and may be changed/removed in the future
                    # in favor of a general-purpose data buffer pool
                    if hasattr(self.train_dataset, "on_batch_end"):
                        # The dataset may be changed after each training batch
                        self.train_dataset.on_batch_end(batch=batch)

                    timing_raw = defaultdict(float)
                    batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

        finally:
            if original_compute_advantage is not None:
                ray_trainer_module.compute_advantage = original_compute_advantage
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].compute_advantage = original_compute_advantage
            if original_compute_reward is not None:
                ray_trainer_module.compute_reward = original_compute_reward
            if original_tracking is not None:
                ray_trainer_module.Tracking = original_tracking
                if "verl.trainer.ppo.ray_trainer" in sys.modules:
                    sys.modules["verl.trainer.ppo.ray_trainer"].Tracking = original_tracking
            if original_tracking_utils is not None:
                tracking_module.Tracking = original_tracking_utils
                if "verl.utils.tracking" in sys.modules:
                    sys.modules["verl.utils.tracking"].Tracking = original_tracking_utils

    def _inject_loss_context(self, batch: Any) -> Any:
        _inject_standard_composer_context(batch)
        if hasattr(self, "composer_config_dict"):
            composer_cfg = dict(self.composer_config_dict)
            meta_info = getattr(batch, "meta_info", None)

            if meta_info is None:
                try:
                    batch.meta_info = {}
                    meta_info = batch.meta_info
                except Exception:
                    meta_info = None

            injected = False
            if isinstance(meta_info, dict):
                meta_info["composer_config"] = composer_cfg
                meta_info["composer_config_json"] = json.dumps(composer_cfg)
                # Primitive backup keys for paths that strip nested dict payloads.
                for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                    if k in composer_cfg:
                        meta_info[f"composer_{k}"] = composer_cfg[k]
                # Inject rollout_n so the worker can reconstruct uid grouping.
                try:
                    _rollout_n = self.config.actor_rollout_ref.rollout.n
                    meta_info["rollout_n"] = int(_rollout_n)
                except Exception:
                    pass
                injected = True
            else:
                try:
                    meta_info["composer_config"] = composer_cfg
                    meta_info["composer_config_json"] = json.dumps(composer_cfg)
                    for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                        if k in composer_cfg:
                            meta_info[f"composer_{k}"] = composer_cfg[k]
                    try:
                        _rollout_n = self.config.actor_rollout_ref.rollout.n
                        meta_info["rollout_n"] = int(_rollout_n)
                    except Exception:
                        pass
                    injected = True
                except Exception:
                    try:
                        batch.meta_info = dict(meta_info) if meta_info is not None else {}
                        batch.meta_info["composer_config"] = composer_cfg
                        batch.meta_info["composer_config_json"] = json.dumps(composer_cfg)
                        for k in ("clip_mode", "agg_mode", "regularizer", "reg_coef"):
                            if k in composer_cfg:
                                batch.meta_info[f"composer_{k}"] = composer_cfg[k]
                        try:
                            _rollout_n = self.config.actor_rollout_ref.rollout.n
                            batch.meta_info["rollout_n"] = int(_rollout_n)
                        except Exception:
                            pass
                        injected = True
                    except Exception:
                        injected = False

            if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
                keys = []
                current_meta = getattr(batch, "meta_info", None)
                if isinstance(current_meta, dict):
                    keys = list(current_meta.keys())
                print(
                    f"[composer-debug] Inject composer_config into batch.meta_info: "
                    f"success={injected}, meta_info_type={type(current_meta)}, meta_info_keys={keys}"
                )

        return batch

    def _update_actor(self, batch):  # type: ignore[override]
        batch = self._inject_loss_context(batch)
        self._validate_actor_batch_contract(batch)
        return super()._update_actor(batch)


def patch_verl_main_ppo() -> None:
    """Patch VERL main PPO wiring to use ComposerRayPPOTrainer and compute_advantage."""

    if ray_trainer_module is None:
        raise RuntimeError(
            "patch_verl_main_ppo requires `verl` to be installed. "
            f"Original import error: {_VERL_IMPORT_ERROR!r}"
        )

    global _ORIGINAL_COMPUTE_ADVANTAGE
    global _ORIGINAL_RAY_TRAINER_CLASS
    global _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS

    if _ORIGINAL_COMPUTE_ADVANTAGE is None:
        _ORIGINAL_COMPUTE_ADVANTAGE = ray_trainer_module.compute_advantage
        ray_trainer_module.compute_advantage = composer_compute_advantage

    if _ORIGINAL_RAY_TRAINER_CLASS is None:
        _ORIGINAL_RAY_TRAINER_CLASS = ray_trainer_module.RayPPOTrainer
        ray_trainer_module.RayPPOTrainer = ComposerRayPPOTrainer

    import verl.trainer.main_ppo as main_ppo

    if _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS is None:
        _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = ComposerRayPPOTrainer

    _patch_dp_actor_update_policy()
    _patch_dp_actor_forward_microbatch_compute_log_prob()
    _patch_fsdp_compute_log_probs()


def unpatch_verl_main_ppo() -> None:
    """Restore VERL's original trainer wiring if it was patched."""

    if ray_trainer_module is None:
        return

    global _ORIGINAL_COMPUTE_ADVANTAGE
    global _ORIGINAL_RAY_TRAINER_CLASS
    global _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS

    if _ORIGINAL_COMPUTE_ADVANTAGE is not None:
        ray_trainer_module.compute_advantage = _ORIGINAL_COMPUTE_ADVANTAGE
        _ORIGINAL_COMPUTE_ADVANTAGE = None

    if _ORIGINAL_RAY_TRAINER_CLASS is not None:
        ray_trainer_module.RayPPOTrainer = _ORIGINAL_RAY_TRAINER_CLASS
        _ORIGINAL_RAY_TRAINER_CLASS = None

    try:
        import verl.trainer.main_ppo as main_ppo

        if _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS is not None:
            main_ppo.RayPPOTrainer = _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS
            _ORIGINAL_MAIN_PPO_RAY_TRAINER_CLASS = None
    except Exception:
        pass

    unpatch_dp_actor_update_policy()
