# Dataset compatibility matrix

This folder contains **dataset overlay YAMLs**. Each one sets the data
loader knobs (batch size, max lengths) and the reward manager appropriate
for that dataset — independent of the algorithm choice.

## How to use

Three equivalent ways:

```yaml
# (a) inside a recipe — Hydra defaults list
defaults:
  - data: gsm8k     # picks configs/data/gsm8k.yaml
  - _self_

algorithm:
  adv_estimator: kalman_grpo
# ... rest of composer block
```

```bash
# (b) from CLI — overlay at launch time
python scripts/train_local.py \
  --config examples/kalman_dapo.yaml \
  --extra-overrides "++data.preset=math ++data.max_response_length=2048"
```

```bash
# (c) explicit parquet paths — bypass presets entirely
python scripts/train_local.py \
  --config examples/kalman_dapo.yaml \
  --train-files /tmp/my/train.parquet --val-files /tmp/my/val.parquet
```

## Dataset → composer compatibility matrix

The columns are the composer axes that may need attention for that dataset.
**"Single"** = single scalar reward, no special composer setup needed.
**"Custom"** = needs a composer reward in `composer_reward_pipeline` or a
non-default `algorithm.adv_estimator`.
**"Need infra"** = requires reward model or multi-signal infra not yet wired.

| Dataset | Reward shape | Wired in this repo | Needs custom advantage? | Needs custom reward pipeline? | Needs reference rewards? | Needs trained RM? | Needs strata ids? |
|---|---|---|---|---|---|---|---|
| **GSM8K** | binary 0/1 (string match) | ✅ `gsm8k.yaml` | no | no | no | no | no |
| **MATH / MATH-Hard** | binary 0/1 (boxed compare) | ✅ `math.yaml` | no | no | no | no | no |
| **DAPO-Math-17K** | binary 0/1 (boxed compare) | ✅ `dapo_math_17k.yaml` | no | optional `length_dependent` | no | no | no |
| **AIME 2024** | binary 0/1 (integer match) | ✅ `aime.yaml` | no | no | no | no | no |
| **AMC** | binary 0/1 (boxed) | ✅ `amc.yaml` | no | no | no | no | no |
| **GPQA** | binary 0/1 (multiple choice letter) | ✅ `gpqa.yaml` | no | no | no | no | no |
| **HumanEval** | binary 0/1 (test-pass) | ✅ `humaneval.yaml` ⚠️ | no | no | no | needs `prime` reward manager | no |
| **MBPP (sanitized)** | binary 0/1 (test-pass) | ✅ `mbpp.yaml` ⚠️ | no | no | no | needs `prime` reward manager | no |
| **CodeContests** | binary 0/1 (stdin/stdout, slow) | ✅ `code_contests.yaml` ⚠️ | no | no | no | needs `prime` reward manager | no |
| **LiveCodeBench** | binary 0/1 (hidden tests) | ✅ `livecodebench.yaml` ⚠️ | no | no | no | needs `prime` reward manager | no |
| **TLDR / OpenAI summarisation** | scalar from RM | ❌ needs trained RM | no | no | no | **yes** | no |
| **HH-RLHF / preference data** | pairwise (chosen/rejected) | ❌ needs RM training first | no | no | no | **yes** | no |
| **Multi-turn search-R1 / RAG** | trajectory-level | ❌ needs async rollout patch | maybe `stratified_grpo` | no | no | no | **yes** (turn count) |
| **GSM8K + MATH mixture** | per-source binary | ✅ feed both parquets | no | optional `multi_reward` | no | no | optional |

⚠️ = the dataset is wired in `prepare_dataset.py`, but evaluation requires a
**code-execution reward manager** (`prime` is the veRL default). Without one,
all rollouts score 0 and group methods will produce zero-variance advantages.

## Which composer recipes assume what

| Recipe / paper | Minimum dataset shape | Why |
|---|---|---|
| Vanilla GRPO, Dr.GRPO, KRPO, λ-GRPO, TIC, TR-GRPO | **single scalar** | Use group statistics over a single reward signal |
| DAPO | **single scalar** | Filter on per-prompt std; works on any binary signal |
| GRPO-LEAD, DARO | **single scalar + difficulty signal** | Difficulty inferred from rollout success rate — no extra column |
| Rank-GRPO, Rewarding-Unlikely | **single scalar** | Re-ranks within group; signal stays single |
| Stratified-GRPO | **single scalar + stratum id** | Needs `non_tensor_batch["stratum"]`; falls back to one stratum if missing |
| MS-GRPO | **single scalar + multi-rollout-scale** | Needs hierarchical rollout config |
| PVPO / GAPO with `composer_flow` | **single scalar + reference reward** | Reference-reward FlowPlugin attaches a baseline; ground-truth used as ref |
| Info-GRPO | **single scalar + augmented prompt pairs** | InfoGRPO FlowPlugin pairs each prompt with a perturbed version |
| Multi-reward composer pipelines | **multiple scalars** | Needs >1 reward column or reward managers stacked |
| Posterior-GRPO with trained RM | **scalar + RM logits** | Needs separate reward worker |

## What the columns map to in code

* "Custom advantage" → `algorithm.adv_estimator: <kalman_grpo|stratified_grpo|...>` in YAML.
* "Custom reward pipeline" → `composer.composer_reward_pipeline: [...]`.
* "Reference rewards" → `algorithm.composer_flow: pvpo` (or `gapo`); the
  `ReferenceRewardFlowPlugin` reads `ground_truth` as the reference signal.
* "Trained RM" → `reward_model.enable: true` + `reward_model.model.path: ...`
  in YAML; needs a separate Ray worker pool.
* "Strata ids" → must be present as a `non_tensor_batch` column at data
  preprocessing time.

## Adding a new dataset preset

1. Add a `_prepare_<name>_dataset(output_dir)` function in
   [`scripts/prepare_dataset.py`](../../scripts/prepare_dataset.py) that
   produces `train.parquet` + `val.parquet` with at minimum:

   ```python
   {
       "data_source": "<name>",
       "prompt": [{"role": "user", "content": "..."}],
       "ability": "<math|code|...>",
       "reward_model": {"style": "rule", "ground_truth": "..."},
       "extra_info": {"split": "...", "index": ...},
   }
   ```

2. Wire it into `_prepare_dataset()` under a new preset string.
3. Add a `configs/data/<name>.yaml` overlay with the right `train_batch_size`,
   `max_prompt_length`, `max_response_length`, and `reward_manager`.
4. Pass `--dataset-preset <name>` (or include the overlay in your recipe).

That's it — every composer recipe now works on the new dataset for free.
