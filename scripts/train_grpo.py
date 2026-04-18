"""
train_grpo.py — veRL GRPO training with grpo_composer components

This is the entry point that:
1. Imports grpo_composer.integrations.verl (triggers registration of all
   custom advantage estimators + the "composer" loss into veRL's registries)
2. Launches veRL's GRPO training loop with the specified config via the
   composer entrypoint, which substitutes ComposerTaskRunner +
   ComposerRayPPOTrainer + ComposerActorRolloutRefWorker into the launch
   path. No global monkey-patching of verl modules is performed.

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
from grpo_composer.integrations.verl import (  # noqa: F401
    aggregations_registery,
    clip_registery,
    regularisation_registery,
    rewards_registery,
    utils,
)
from grpo_composer.integrations.verl.entrypoint import run as composer_run


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


# ────────────────────────────────────────────────────
# Step 2: Hydra entrypoint via veRL's main()
# ────────────────────────────────────────────────────
# veRL's `main()` Hydra-loads the config and calls `run_ppo(config)`. We
# redirect `run_ppo` to our composer entrypoint so ComposerTaskRunner is the
# Ray actor class. This is a single launch-time seam, not a class table
# monkey-patch — all behavioral overrides live on the composer subclasses.

import verl.trainer.main_ppo as _main_ppo  # noqa: E402


def _composer_run_ppo(config, task_runner_class=None):  # noqa: ARG001
    _inject_composer_env_into_ray_runtime(config)
    composer_run(config)


_main_ppo.run_ppo = _composer_run_ppo

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    print("=" * 60)
    print("  grpo_composer components registered into veRL")
    print("=" * 60)
    print()
    print("  Advantages:  difficulty_aware, length_corrected, kalman,")
    print("               decoupled, multi_scale, static_value, novelty_sharp,")
    print("               unbiased_grpo")
    print("  Loss:        composer (clip_mode × agg_mode × regularizer)")
    print("  Trainer:     ComposerRayPPOTrainer (via ComposerTaskRunner)")
    print()
    print("=" * 60)

    main()
