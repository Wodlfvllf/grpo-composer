# Tutorial: Creating Custom Components

## 1. Implement Core Logic

Add your implementation under the correct family in `grpo_composer/core/`:

- advantage estimator
- clipping mechanism
- aggregation strategy
- regularizer
- reward transform

## 2. Integrate with veRL

Register your component in the veRL integration layer:

- `grpo_composer/integrations/verl/advantages.py`
- `grpo_composer/integrations/verl/losses.py`
- `grpo_composer/integrations/verl/trainer.py` (if trainer-side plumbing is needed)

## 3. Add Config Surface

Expose config keys in YAML under `configs/` and verify launcher-generated overrides set the expected fields.

## 4. Add Tests

Add targeted tests under `tests/` for:

- math correctness
- shape and dtype safety
- failure mode clarity

## 5. Validate via Smoke Run

Use one official launcher:

- `scripts/train_modal.py` (preferred)
- `scripts/train_local.py` (fallback)

Preflight sanity checks should pass before training starts.
