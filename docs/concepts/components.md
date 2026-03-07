# Components

`grpo_composer` is organized as composable RL components plus a veRL integration layer.

## Core Component Families

Under `grpo_composer/core/`:

- `advantages/`: GRPO estimator variants
- `clipping/`: clipping mechanisms
- `aggregation/`: token/trajectory/group aggregation
- `regularizers/`: KL and related regularization terms
- `rewards/`: reward transformation modules

## veRL Integration Layer

Under `grpo_composer/integrations/verl/`:

- `advantages.py`: registers custom advantage estimators
- `losses.py`: registers `composer` loss mode
- `trainer.py`: patches veRL trainer entrypoints for composer-aware behavior

## Runtime Entry

Training entrypoint:

- [`scripts/train_grpo.py`](../../scripts/train_grpo.py)

Recommended launchers:

- [`scripts/train_modal.py`](../../scripts/train_modal.py)
- [`scripts/train_local.py`](../../scripts/train_local.py)

These launchers add sane defaults and run preflight checks before invoking `train_grpo.py`.
