# API Reference (Minimal)

## Runtime

- `grpo_composer.runtime_stack.CANONICAL_RUNTIME`
- `grpo_composer.runtime_stack.CANONICAL_PIP_PACKAGES`
- `grpo_composer.runtime_stack.validate_runtime_stack()`

## Config Sanity

- `grpo_composer.config.build_effective_config(base_cfg, overrides)`
- `grpo_composer.config.run_preflight_sanity_checks(config_path, overrides, train_file)`

## Launchers

- `scripts/train_modal.py` (canonical)
- `scripts/train_local.py` (local fallback)

## veRL Integration

- `grpo_composer.integrations.verl.advantages`
- `grpo_composer.integrations.verl.losses`
- `grpo_composer.integrations.verl.trainer`

## Notes

Legacy launch shell scripts are retained only for compatibility and are not canonical.
