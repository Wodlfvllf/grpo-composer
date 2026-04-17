from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launcher_common as lc


def test_build_train_grpo_command() -> None:
    assert lc.build_train_grpo_command(["++foo=1"]) == [
        "python",
        "scripts/train_grpo.py",
        "++foo=1",
    ]


def test_build_training_overrides_includes_expected_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "composer:\n"
        "  clip_mode: symmetric\n"
        "  reg_coef: 0.1\n",
        encoding="utf-8",
    )

    overrides = lc.build_training_overrides(
        config_path=config_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_files="/tmp/train.parquet",
        val_files="/tmp/val.parquet",
        run_name="smoke_run",
        total_epochs=3,
        n_gpus_per_node=1,
        checkpoint_dir=Path("/checkpoints/smoke_run"),
        project_name="grpo_composer_modal",
        default_logger=["console", "wandb"],
        extra_overrides="",
    )

    assert '++composer.clip_mode="symmetric"' in overrides
    assert "++composer.reg_coef=0.1" in overrides
    assert "++trainer.project_name=grpo_composer_modal" in overrides
    assert '++trainer.logger=["console","wandb"]' in overrides
    assert "++actor_rollout_ref.rollout.gpu_memory_utilization=0.5" in overrides
    assert "++actor_rollout_ref.model.external_lib=grpo_composer.integrations.verl" in overrides


def test_build_training_overrides_respects_existing_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("trainer:\n  default_local_dir: /tmp/ignored\n", encoding="utf-8")

    extra = (
        "++trainer.logger=[\"console\"] "
        "++actor_rollout_ref.rollout.max_model_len=4096 "
        "++actor_rollout_ref.actor.ppo_micro_batch_size=4 "
        "++critic.model.override_config.attn_implementation=flash_attention_2"
    )

    overrides = lc.build_training_overrides(
        config_path=config_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_files="/tmp/train.parquet",
        val_files="/tmp/val.parquet",
        run_name="smoke_run",
        total_epochs=3,
        n_gpus_per_node=4,
        checkpoint_dir=Path("/checkpoints/smoke_run"),
        project_name="grpo_composer_local",
        default_logger=["console"],
        extra_overrides=extra,
    )

    assert "++trainer.logger=[console]" in overrides
    assert "++actor_rollout_ref.rollout.max_model_len=4096" in overrides
    assert "++actor_rollout_ref.actor.ppo_micro_batch_size=4" in overrides
    assert "++critic.model.override_config.attn_implementation=flash_attention_2" in overrides

    assert "++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8" not in overrides
    assert "++actor_rollout_ref.rollout.max_model_len=2048" not in overrides
    assert "++critic.model.override_config.attn_implementation=sdpa" not in overrides
    assert "++critic.model.override_config._attn_implementation=sdpa" not in overrides


def test_build_launcher_env_defaults_and_pythonpath() -> None:
    env = lc.build_launcher_env(
        Path("/repo"),
        base_env={
            "PYTHONPATH": "existing:path",
            "TOKENIZERS_PARALLELISM": "true",
        },
    )

    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["TOKENIZERS_PARALLELISM"] == "true"
    assert env["PYTHONPATH"] == "/repo:existing:path"
    assert env["VERL_LOG_LEVEL"] == "WARNING"
    assert env["RAY_DEDUP_LOGS"] == "1"
    assert env["RAY_IGNORE_UNHANDLED_ERRORS"] == "1"
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["VLLM_LOGGING_LEVEL"] == "ERROR"
