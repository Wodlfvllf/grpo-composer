from __future__ import annotations
import sys
from pathlib import Path

# Add mounted repo to Python path
REPO_ROOT = Path("/root/grpo_composer")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.modal_app import (
    app,
    image,
    checkpoint_volume,
    GPU_CONFIG,
    TIMEOUT_SEC,
    CHECKPOINT_ROOT,
)
from scripts.training_runner import run_training_pipeline
# from training_runner import run_training_pipeline 


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT_SEC,
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def run_training(**kwargs):
    return run_training_pipeline(
        **kwargs,
        remote_root=REPO_ROOT,
        checkpoint_root=CHECKPOINT_ROOT,
        wandb_enabled=True,
    )


@app.local_entrypoint()
def main(
    config: str = "configs/base_grpo.yaml",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_files: str = "",
    val_files: str = "",
    dataset_preset: str = "gsm8k",
    run_name: str = "grpo_modal_run",
    total_epochs: int = 3,
    n_gpus_per_node: int = 1,
    extra_overrides: str = "",
    debug: bool = False,
):
    run_training.remote(
        config=config,
        model=model,
        train_files=train_files,
        val_files=val_files,
        dataset_preset=dataset_preset,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        extra_overrides=extra_overrides,
        debug=debug,
    )