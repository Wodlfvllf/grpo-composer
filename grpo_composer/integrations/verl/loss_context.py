"""Shared config/context state for veRL composer loss integration."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from verl.workers.config import ActorConfig

_COMPOSER_CONFIG: dict[str, Any] = {}
_COMPOSER_CONFIG_LOADED: bool = False
_COMPOSER_BATCH_CONTEXT: dict[str, Any] = {}


def set_composer_config(config_dict: dict[str, Any]) -> None:
    """Store composer config globally in the current worker process."""
    global _COMPOSER_CONFIG, _COMPOSER_CONFIG_LOADED
    _COMPOSER_CONFIG = dict(config_dict or {})
    _COMPOSER_CONFIG_LOADED = True


def _ensure_composer_config_loaded() -> None:
    global _COMPOSER_CONFIG, _COMPOSER_CONFIG_LOADED
    if _COMPOSER_CONFIG_LOADED:
        return

    _COMPOSER_CONFIG_LOADED = True
    raw = os.environ.get("GRPO_COMPOSER_CONFIG")
    if not raw:
        return

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            _COMPOSER_CONFIG = parsed
    except Exception:
        pass


def get_composer_config() -> dict[str, Any]:
    _ensure_composer_config_loaded()
    return dict(_COMPOSER_CONFIG)


def set_composer_batch_context(context: dict[str, Any]) -> None:
    global _COMPOSER_BATCH_CONTEXT
    _COMPOSER_BATCH_CONTEXT = dict(context or {})


def get_composer_batch_context() -> dict[str, Any]:
    return dict(_COMPOSER_BATCH_CONTEXT)


def clear_composer_batch_context() -> None:
    global _COMPOSER_BATCH_CONTEXT
    _COMPOSER_BATCH_CONTEXT = {}


def config_get(
    config: Optional[ActorConfig], key: str, default, composer_dict: Optional[dict[str, Any]] = None
):
    # 1) Explicit composer dict (strongest signal for custom loss settings)
    if isinstance(composer_dict, dict) and key in composer_dict and composer_dict[key] is not None:
        return composer_dict[key]

    # 2) Module-level composer config (set in-worker or loaded from env)
    _ensure_composer_config_loaded()
    if key in _COMPOSER_CONFIG and _COMPOSER_CONFIG[key] is not None:
        return _COMPOSER_CONFIG[key]

    # 3) ActorConfig / FSDPActorConfig (veRL-native keys)
    if config is not None:
        getter = getattr(config, "get", None)
        if callable(getter):
            val = getter(key, None)
            if val is not None:
                return val
        val = getattr(config, key, None)
        if val is not None:
            return val

    return default


def config_get_context(
    config: Optional[ActorConfig], key: str, default=None, composer_dict: Optional[dict[str, Any]] = None
):
    return config_get(config, key, default, composer_dict)
