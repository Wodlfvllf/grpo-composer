from __future__ import annotations

import os
from pathlib import Path
import modal

from scripts.runtime_stack import CANONICAL_PIP_PACKAGES

APP_NAME = "grpo-composer-train"
REMOTE_ROOT = Path("/root/grpo_composer")
CHECKPOINT_ROOT = Path("/checkpoints")
DATA_ROOT = Path("/tmp/grpo_data")

def _parse_gpu_config(value: str) -> str | list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "A100-80GB"
    return parts


def _get_secret(secret_name: str) -> modal.Secret | None:
    if not secret_name:
        return None
    return modal.Secret.from_name(secret_name)


# -------- Runtime config (infra only) --------
GPU_CONFIG = _parse_gpu_config(os.environ.get("MODAL_GPU", "A100-80GB"))
TIMEOUT_SEC = int(os.environ.get("MODAL_TIMEOUT_SEC", str(24 * 60 * 60)))
MAX_RETRIES = int(os.environ.get("MODAL_MAX_RETRIES", "0"))

CHECKPOINT_VOLUME_NAME = os.environ.get(
    "MODAL_CHECKPOINT_VOLUME",
    "grpo-composer-checkpoints",
)

HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "").strip()
WANDB_SECRET_NAME = os.environ.get("MODAL_WANDB_SECRET_NAME", "").strip()

WANDB_ENTITY_NAME = (
    os.environ.get("WANDB_ENTITY", "").strip()
    or os.environ.get("MODAL_WANDB_ENTITY", "").strip()
)
WANDB_PROJECT_NAME = (
    os.environ.get("WANDB_PROJECT", "").strip()
    or os.environ.get("MODAL_WANDB_PROJECT", "").strip()
)

_secrets = [
    s for s in (
        _get_secret(HF_SECRET_NAME),
        _get_secret(WANDB_SECRET_NAME)
    ) if s is not None
]

# -------- Modal image --------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "MODAL_HF_SECRET_NAME": HF_SECRET_NAME,
            "MODAL_WANDB_SECRET_NAME": WANDB_SECRET_NAME,
            "WANDB_ENTITY": WANDB_ENTITY_NAME,
            "WANDB_PROJECT": WANDB_PROJECT_NAME,
        }
    )
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install(*CANONICAL_PIP_PACKAGES)
    .add_local_dir(".", remote_path=str(REMOTE_ROOT), copy=True)
)

app = modal.App(APP_NAME)

checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME, create_if_missing=True
)