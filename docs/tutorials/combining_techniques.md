# Tutorial: Combining Techniques

This repository supports composition by setting multiple config knobs in one YAML.

## Recommended Workflow

1. Start from `configs/base_grpo.yaml`.
2. Add one technique at a time (advantage, regularizer, reward transform).
3. Launch via `scripts/train_modal.py` or `scripts/train_local.py`.
4. Resolve any preflight errors before scaling runs.

## Example

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: composer

composer:
  agg_mode: token_mean
  clip_mode: symmetric
  clip_ratio: 0.2
  regularizer: kl
  reg_coef: 0.01
```

## Important Guardrails

- `actor_rollout_ref.actor.policy_loss.loss_mode` must be `composer`.
- Some estimators/pipelines require extra dataset columns in parquet files.
- Info-GRPO-style mutual information paths need extra rollout tensors not present in the default launcher path.
