# Tutorial: Reproducing Paper Variants

Paper configs live under `configs/papers/`.

## Practical Reproduction Steps

1. Start from a paper YAML in `configs/papers/`.
2. Run with canonical launcher `scripts/train_modal.py`.
3. Keep `--total-epochs 1` for first smoke run.
4. Confirm checkpoints and step metrics are produced.
5. Scale batch/model/hardware only after successful smoke validation.

## Readiness Notes

- Many paper variants run with existing plumbing.
- Some require additional dataset signals (extra parquet columns).
- Some advanced flows may require extra rollout tensors beyond default launchers.

For a detailed readiness matrix and gaps, see:

- [`docs/repository_state_and_experiment_guide.md`](../repository_state_and_experiment_guide.md)
