from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from scripts.runtime_stack import (
    runtime_summary_text,
    validate_runtime_stack,
)

from scripts.launcher_common import composer_yaml_to_overrides, pkg_version

from scripts.training_config import (
    prepare_dataset,
    build_training_config,
    build_command,
    build_env,
)


def log_runtime_versions():
    print(
        "Runtime versions:",
        {
            "verl": pkg_version("verl"),
            "vllm": pkg_version("vllm"),
            "ray": pkg_version("ray"),
            "transformers": pkg_version("transformers"),
            "torch": pkg_version("torch"),
        },
    )


def execute_training(command, env, cwd):
    print("Launching training command:")
    print(" ".join(shlex.quote(item) for item in command))

    subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=True,
    )


def _read_yaml_data_preset(config_path: Path) -> str:
    """Best-effort read of `data.preset` from a composer YAML.

    Returns "" if the field is missing, the file is unreadable, or the value
    is not a non-empty string. Never raises — the launcher should still work
    even if the YAML lacks a data block.
    """
    try:
        import yaml  # local import; pyyaml is a hard dependency anyway
        with config_path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return ""
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, dict):
        return ""
    preset = data.get("preset")
    return preset.strip() if isinstance(preset, str) and preset.strip() else ""


def _dataset_overlay_overrides(repo_root: Path, dataset_preset: str) -> list[str]:
    """Flatten configs/data/<preset>.yaml into Hydra ++key=value overrides.

    Returns [] silently if no overlay exists for the preset (so the launcher
    keeps working with bare presets that have no overlay yet). The overlay
    keys are emitted *before* the recipe-derived overrides, so the recipe
    YAML and any --extra-overrides win on conflicts (Hydra: last `++` wins).
    """
    if not dataset_preset:
        return []
    # Try a few canonical name variants so 'math', 'math-hard', 'lighteval'
    # all resolve to configs/data/math.yaml and so on.
    candidates = [dataset_preset, dataset_preset.replace("-", "_"), dataset_preset.lower()]
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        overlay = repo_root / "configs" / "data" / f"{name}.yaml"
        if overlay.exists():
            try:
                return composer_yaml_to_overrides(overlay)
            except Exception as exc:
                print(f"[launcher] WARNING: could not read dataset overlay {overlay}: {exc!r}")
                return []
    return []


def run_training_pipeline(
    *,
    config: str,
    model: str,
    train_files: str,
    val_files: str,
    dataset_preset: str,
    run_name: str,
    total_epochs: int,
    n_gpus_per_node: int,
    extra_overrides: str,
    debug: bool,
    remote_root: Path,
    checkpoint_root: Path,
    wandb_enabled: bool,
):
    validate_runtime_stack()

    if not run_name:
        raise ValueError("run_name is required")

    config_path = (remote_root / config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # If the YAML carries data.preset, let it override the CLI default. This
    # lets a recipe pin its expected dataset (e.g. configs/data/math.yaml)
    # without forcing the launcher caller to remember --dataset-preset.
    yaml_preset = _read_yaml_data_preset(config_path)
    if yaml_preset and dataset_preset in ("", "gsm8k"):
        # Only override the CLI default ('' or the launcher's gsm8k default);
        # an explicit non-default --dataset-preset always wins.
        dataset_preset = yaml_preset

    # ---- pipeline ----
    train_files, val_files = prepare_dataset(
        train_files, val_files, dataset_preset
    )

    overrides = build_training_config(
        config_path=config_path,
        model=model,
        train_files=train_files,
        val_files=val_files,
        run_name=run_name,
        total_epochs=total_epochs,
        n_gpus_per_node=n_gpus_per_node,
        checkpoint_dir=checkpoint_dir,
        extra_overrides=extra_overrides,
        wandb_enabled=wandb_enabled,
    )

    # Prepend the dataset overlay (configs/data/<preset>.yaml) so its keys
    # (e.g. data.truncation, data.max_prompt_length, reward_model.reward_manager)
    # actually reach veRL. Recipe YAML and --extra-overrides come after in
    # `overrides` so they win on any key conflict (Hydra: last ++ wins).
    overlay_overrides = _dataset_overlay_overrides(remote_root, dataset_preset)
    if overlay_overrides:
        print(f"[launcher] applying dataset overlay configs/data/{dataset_preset}.yaml "
              f"({len(overlay_overrides)} keys)")
        overrides = overlay_overrides + overrides

    if debug:
        # Ray actors do not inherit the driver's env, so push the DAPO debug
        # flag through OmegaConf (which is serialised into the
        # ComposerTaskRunner / ComposerRayPPOTrainer actor process) instead of
        # relying solely on env vars. _dapo_debug_enabled prefers the config
        # value over GRPO_COMPOSER_DAPO_DEBUG.
        overrides.append("++algorithm.filter_groups.debug=true")

    command = build_command(overrides)

    log_runtime_versions()

    env = build_env(remote_root)

    if debug:
        env["GRPO_COMPOSER_DEBUG"] = "1"
        env["GRPO_COMPOSER_DAPO_DEBUG"] = "1"
        env["GRPO_COMPOSER_STRICT_VALIDATION"] = "1"

    execute_training(command, env, remote_root)

    return str(checkpoint_dir)