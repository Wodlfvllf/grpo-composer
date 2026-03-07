# GRPO Composer Docs

This docs set defines the **canonical runtime and launcher path** for this repository.

## Canonical Runtime Stack

- Python: `3.11`
- veRL: `0.6.1`
- vLLM: `0.8.5`
- Ray: `>=2.40.0`
- Transformers: `>=4.51.0,<5.0.0`
- Tensordict: `>=0.8.0,<=0.10.0,!=0.9.0`

Source of truth: [`grpo_composer/runtime_stack.py`](../grpo_composer/runtime_stack.py)

## Official Launchers

1. **Canonical production path**: [`scripts/train_modal.py`](../scripts/train_modal.py)
2. **Official local fallback**: [`scripts/train_local.py`](../scripts/train_local.py)

Legacy wrappers:

- [`scripts/train.sh`](../scripts/train.sh) (deprecated wrapper to local launcher)
- [`scripts/train_multinode.sh`](../scripts/train_multinode.sh) (deprecated and disabled)

## Preflight Safety

Both official launchers run fail-fast checks from [`grpo_composer/config/sanity.py`](../grpo_composer/config/sanity.py):

- validates composer loss mode wiring
- blocks unsupported mode combinations
- validates required dataset signals for certain estimators/pipelines
- fails before expensive rollout/training starts

## Read Next

- [Getting Started](./getting_started.md)
- [Concepts: Components](./concepts/components.md)
- [Tutorials: Reproducing Papers](./tutorials/reproducing_papers.md)
- [API Reference](./api/reference.md)
- [Repository State and Experiment Guide](./repository_state_and_experiment_guide.md)
