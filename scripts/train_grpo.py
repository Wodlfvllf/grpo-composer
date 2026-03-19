"""
train_grpo.py — veRL GRPO training with grpo_composer components

This is the entry point that:
1. Imports grpo_composer.integrations.verl (triggers registration of all
   custom advantage estimators + the "composer" loss into veRL's registries)
2. Launches veRL's GRPO training loop with the specified config

Usage:
    # Single GPU (for debugging):
    python scripts/train_grpo.py --config configs/base_grpo.yaml --model.path Qwen/Qwen2.5-1.5B

    # Multi-GPU via torchrun:
    torchrun --nproc_per_node=4 scripts/train_grpo.py --config configs/dapo_7b.yaml

    # Or via the train.sh wrapper:
    bash scripts/train.sh configs/custom_mix.yaml Qwen/Qwen2.5-7B openai/gsm8k 4
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Ensure local repository root is first on sys.path when this file is executed
# as a script (e.g., in Modal), so imports resolve to the mounted workspace code.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ────────────────────────────────────────────────────
# Step 1: Register grpo_composer components into veRL
# ────────────────────────────────────────────────────
# This single import triggers all @register_adv_est and @register_policy_loss
# decorators, making our custom components available to veRL.

import grpo_composer.integrations.verl  # noqa: F401  — side-effect import
from grpo_composer.integrations.verl import patch_verl_main_ppo
from grpo_composer.integrations.verl import (  # noqa: F401
    aggregations_registery,
    clip_registery,
    patch_dp_actor,
    regularisation_registery,
    rewards_registery,
    utils,
)


# ────────────────────────────────────────────────────
# Step 2: Custom TaskRunner that re-patches inside
#         the Ray actor process (Process 3)
# ────────────────────────────────────────────────────
# Ray spawns TaskRunner as a separate process. Monkey-patches
# from the main process (Process 2) don't carry over. This
# subclass re-applies them so ComposerRayPPOTrainer and custom
# advantage estimators are available inside the Ray actor.

patch_verl_main_ppo()

import ray
import verl.trainer.main_ppo as _main_ppo
from verl.trainer.main_ppo import TaskRunner


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except Exception:
            pass
    try:
        value = obj[key]
        return default if value is None else value
    except Exception:
        pass
    value = getattr(obj, key, default)
    return default if value is None else value


def _to_plain_dict(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass

    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _inject_composer_env_into_ray_runtime(config: Any) -> None:
    """Ensure all Ray actors/workers start with composer config in env.

    This is the most reliable cross-process transport for strict veRL dataclass
    setups where custom config keys cannot be added under actor configs.
    """

    composer_cfg = _to_plain_dict(_cfg_get(config, "composer", None))
    composer_json = json.dumps(composer_cfg, sort_keys=True) if composer_cfg else None

    # Ray worker processes may not inherit all driver env vars when runtime_env
    # is set. Explicitly forward authentication/env knobs required by integrations.
    passthrough_env: dict[str, str] = {}
    for env_key in (
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_MODE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        env_value = os.environ.get(env_key)
        if env_value:
            passthrough_env[env_key] = env_value

    try:
        from omegaconf import OmegaConf, open_dict

        with open_dict(config):
            if _cfg_get(config, "ray_kwargs", None) is None:
                config.ray_kwargs = OmegaConf.create({})
            with open_dict(config.ray_kwargs):
                if _cfg_get(config.ray_kwargs, "ray_init", None) is None:
                    config.ray_kwargs.ray_init = OmegaConf.create({})
                with open_dict(config.ray_kwargs.ray_init):
                    if _cfg_get(config.ray_kwargs.ray_init, "runtime_env", None) is None:
                        config.ray_kwargs.ray_init.runtime_env = OmegaConf.create({})
                    with open_dict(config.ray_kwargs.ray_init.runtime_env):
                        if _cfg_get(config.ray_kwargs.ray_init.runtime_env, "env_vars", None) is None:
                            config.ray_kwargs.ray_init.runtime_env.env_vars = OmegaConf.create({})
                        with open_dict(config.ray_kwargs.ray_init.runtime_env.env_vars):
                            if composer_json is not None:
                                config.ray_kwargs.ray_init.runtime_env.env_vars["GRPO_COMPOSER_CONFIG"] = composer_json
                            for env_key, env_value in passthrough_env.items():
                                config.ray_kwargs.ray_init.runtime_env.env_vars[env_key] = env_value
    except Exception:
        # Dict-like fallback for non-OmegaConf configs.
        ray_kwargs = _to_plain_dict(_cfg_get(config, "ray_kwargs", {}))
        ray_init = _to_plain_dict(_cfg_get(ray_kwargs, "ray_init", {}))
        runtime_env = _to_plain_dict(_cfg_get(ray_init, "runtime_env", {}))
        env_vars = _to_plain_dict(_cfg_get(runtime_env, "env_vars", {}))
        if composer_json is not None:
            env_vars["GRPO_COMPOSER_CONFIG"] = composer_json
        env_vars.update(passthrough_env)
        runtime_env["env_vars"] = env_vars
        ray_init["runtime_env"] = runtime_env
        ray_kwargs["ray_init"] = ray_init
        try:
            config["ray_kwargs"] = ray_kwargs
        except Exception:
            pass

    # Also set in current process for local fallbacks and eager imports.
    if composer_json is not None:
        os.environ["GRPO_COMPOSER_CONFIG"] = composer_json

    if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
        keys = sorted(composer_cfg.keys()) if composer_cfg else []
        forwarded = sorted(passthrough_env.keys())
        print(
            "[composer-debug] Injected runtime env for Ray workers. "
            f"keys={keys} forwarded_env={forwarded}"
        )


class ComposerTaskRunner(TaskRunner):
    """TaskRunner that ensures grpo_composer patches are applied in this process."""

    def run(self, config):
        # Re-apply patches inside the Ray actor process
        import grpo_composer.integrations.verl  # noqa: F401
        from grpo_composer.integrations.verl import patch_verl_main_ppo
        patch_verl_main_ppo()
        if os.environ.get("GRPO_COMPOSER_DEBUG") == "1":
            print(
                "[composer-debug] TaskRunner env check: "
                f"WANDB_API_KEY={'present' if bool(os.environ.get('WANDB_API_KEY')) else 'missing'} "
                f"WANDB_ENTITY={'present' if bool(os.environ.get('WANDB_ENTITY')) else 'missing'} "
                f"WANDB_PROJECT={'present' if bool(os.environ.get('WANDB_PROJECT')) else 'missing'}"
            )
        return super().run(config)


# Monkey-patch run_ppo so veRL's main() uses our ComposerTaskRunner
_original_run_ppo = _main_ppo.run_ppo


def _composer_run_ppo(config, task_runner_class=None):
    _inject_composer_env_into_ray_runtime(config)
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(ComposerTaskRunner)
    return _original_run_ppo(config, task_runner_class=task_runner_class)


_main_ppo.run_ppo = _composer_run_ppo

from verl.trainer.main_ppo import main


if __name__ == "__main__":
    print("=" * 60)
    print("  grpo_composer components registered into veRL")
    print("=" * 60)
    print()
    print("  Advantages:  difficulty_aware, length_corrected, kalman,")
    print("               decoupled, multi_scale, static_value, novelty_sharp,")
    print("               unbiased_grpo")
    print("  Loss:        composer (clip_mode × agg_mode × regularizer)")
    print("  Trainer:     ComposerRayPPOTrainer (patched over RayPPOTrainer)")
    print()
    print("=" * 60)

    main()
