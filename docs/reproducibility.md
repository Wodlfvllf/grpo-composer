# Reproducibility Tracker

This file tracks which paper configs in [`configs/papers/`](../configs/papers/) have been
empirically validated, on what hardware, and how the achieved metric compares
to the published number.

> **Honesty policy:** rows are filled in only after a real end-to-end run.
> Anything unverified is marked **untested**. PRs adding rows (or correcting
> them) are very welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).

## How to read this table

| Column | Meaning |
| --- | --- |
| **Config** | YAML in `configs/papers/` |
| **Base model** | Pretrained checkpoint passed via `actor_rollout_ref.model.path` |
| **Dataset** | Preset from `configs/data/` |
| **Steps** | `trainer.total_training_steps` |
| **Hardware** | GPU type × count |
| **GPU-hours** | Wall-clock × GPU count |
| **Metric** | Eval metric (dataset-specific: pass@1, exact match, etc.) |
| **Achieved** | What this repo got |
| **Paper** | What the paper reports for the same setting |
| **Run** | Link to W&B run / report |
| **Status** | ✅ matches / ⚠️ partial / ❌ regression / 🟡 in-progress / ❔ untested |

## Validated runs

| Config | Base model | Dataset | Steps | Hardware | GPU-hours | Metric | Achieved | Paper | Run | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dapo.yaml` | _TBD_ | dapo_math_17k | _TBD_ | _TBD_ | _TBD_ | AIME24 pass@1 | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `dr_grpo.yaml` | _TBD_ | math | _TBD_ | _TBD_ | _TBD_ | MATH pass@1 | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `lambda_grpo.yaml` | _TBD_ | gsm8k | _TBD_ | _TBD_ | _TBD_ | GSM8K acc | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `grpo_lead.yaml` | _TBD_ | math | _TBD_ | _TBD_ | _TBD_ | MATH pass@1 | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `stratified_grpo.yaml` | _TBD_ | math | _TBD_ | _TBD_ | _TBD_ | MATH pass@1 | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `rank_grpo.yaml` | _TBD_ | math | _TBD_ | _TBD_ | _TBD_ | MATH pass@1 | _TBD_ | _TBD_ | _TBD_ | ❔ untested |
| `amir_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `daro.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `dra_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `gapo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `gdpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `info_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `krpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `ms_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `posterior_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `pvpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `rewarding_unlikely.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `spo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `tic_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `tr_grpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |
| `xrpo.yaml` |  |  |  |  |  |  |  |  |  | ❔ untested |

## Adding a row

1. Run an end-to-end training job using one of the configs in `configs/papers/`.
2. Note the W&B run URL (or attach a `wandb report` link if you generated one).
3. Replace the placeholder row above with the real numbers, **including any
   deltas from the published result**. Negative results are valuable; please
   don't hide them.
4. Open a PR. If the achieved number diverges materially from the paper, add
   a short note in the run report explaining suspected reasons (different
   base model, fewer steps, dataset filtering, etc.).

## Caveats

- veRL 0.6.1 ships built-in scorers for a subset of the datasets we wire; see
  [`configs/data/README.md`](../configs/data/README.md) for the compatibility
  matrix. Datasets without an upstream scorer (GPQA, HumanEval, MBPP, LCB)
  need a custom reward function before they can produce an "Achieved" number.
- GPU-hours below should be the **total** (wall × world_size), not per-GPU.
