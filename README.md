# grpo-composer

A modular library for **Group Relative Policy Optimization (GRPO)** and its
many descendants, built as a thin composer on top of
[veRL 0.6.1](https://github.com/volcengine/verl).

Instead of forking veRL once per paper, every GRPO variant is expressed as a
**combination of orthogonal components** — advantage estimator, clipping
mode, aggregation, regulariser, reward pipeline — selected via YAML.
Twenty-plus published methods are reproduced with the same code path; new
methods are usually one config file.

```
                       ┌──────────── composer ────────────┐
algorithm.adv_estimator│  advantages/   clipping/   regularizers/ │
composer.clip_mode     │  aggregation/  rewards/    losses/       │
composer.agg_mode      │                                          │
composer.regularizer   └──────────────────────────────────────────┘
                                       │
                                       ▼
                ┌─────── integrations/verl ────────────┐
                │ ComposerDataParallelPPOActor          │
                │ ComposerActorRolloutRefWorker         │
                │ ComposerRayPPOTrainer                 │
                │ ComposerTaskRunner (launch entrypoint)│
                └───────────────────────────────────────┘
                                       │
                                       ▼
                                  veRL 0.6.1
```

## Why

Every paper after GRPO (DAPO, KRPO, λ-GRPO, GRPO-LEAD, Dr. GRPO, DARO,
DRA-GRPO, Stratified-GRPO, …) tweaks a *small subset* of the same
gradient pipeline: how the advantage is estimated, how the ratio is clipped,
how tokens are aggregated, how the policy is regularised, how the reward is
shaped. `grpo-composer` factors those axes apart so a paper becomes a
3–10 line YAML, and a novel hybrid is the same.

No method-table monkey-patching of veRL — composer behaviour is concentrated
in three subclasses + one Hydra launcher (see
[`grpo_composer/integrations/verl/`](grpo_composer/integrations/verl)).

## Repository layout

| Path | Purpose |
|---|---|
| [`grpo_composer/core/`](grpo_composer/core) | The composer building blocks: `advantages/`, `aggregation/`, `clipping/`, `losses/`, `regularizers/`, `rewards/` |
| [`grpo_composer/integrations/verl/`](grpo_composer/integrations/verl) | The veRL plumbing: composer subclasses, flow plugins, registries, helpers |
| [`configs/`](configs) | **Pinned** YAMLs — `base_grpo.yaml` and one file per published paper in [`configs/papers/`](configs/papers) |
| [`examples/`](examples) | **Mutable** mix-and-match recipes — hand-designed novel combinations for experimentation |
| [`scripts/`](scripts) | Launch entrypoints: `train_grpo.py` (Hydra), `train_local.py`, `train_modal.py` |
| [`docs/`](docs) | Getting started + experiment guide |
| [`tests/`](tests) | Unit tests for composer components |
| [`configs/data/`](configs/data) | Dataset overlays (GSM8K, MATH, AIME, AMC, GPQA, DAPO-Math-17K, HumanEval, MBPP, CodeContests, LiveCodeBench) — see [`configs/data/README.md`](configs/data/README.md) |

## Install

```powershell
git clone https://github.com/Wodlfvllf/grpo-composer.git
cd grpo-composer
pip install -e .
# veRL is installed alongside (see pyproject.toml extras)
```

## Quick start

Vanilla GRPO on GSM8K, 1 GPU, locally:

```powershell
python scripts/train_local.py --config configs/base_grpo.yaml
```

A published paper (DAPO, with dynamic sampling):

```powershell
python scripts/train_local.py --config configs/papers/dapo.yaml
```

A novel mix from the cookbook (Kalman baseline + DAPO clip + global-token agg):

```powershell
python scripts/train_local.py --config examples/kalman_dapo.yaml
```

A different dataset (MATH-Hard) with the same recipe:

```powershell
python scripts/train_local.py --config examples/kalman_dapo.yaml --dataset-preset math
```

On Modal:

```powershell
modal run scripts/train_modal.py --config examples/kalman_dapo.yaml --debug
```

`--debug` turns on `GRPO_COMPOSER_DEBUG`, `GRPO_COMPOSER_DAPO_DEBUG`, and
`GRPO_COMPOSER_STRICT_VALIDATION` so you see full composer + DAPO oversampling
diagnostics.

## Implemented methods (papers)

Each lives at `configs/papers/<name>.yaml` and reproduces the paper's
hyper-parameters faithfully.

| Method | Key axis touched |
|---|---|
| GRPO (DeepSeekMath) | baseline |
| Dr. GRPO | normalisation |
| DAPO | asymmetric clip + dynamic sampling |
| DARO | difficulty-weighted aggregation |
| GRPO-LEAD | length-dependent reward + difficulty-aware advantage |
| KRPO | Kalman baseline |
| λ-GRPO | learnable token aggregation |
| TIC-GRPO | length-corrected advantage + trajectory clip |
| TR-GRPO | weighted-trust clip |
| Rank-GRPO | rank-enhanced reward |
| Rewarding Unlikely | unlikeliness reward |
| Posterior-GRPO | posterior-composite reward |
| Stratified-GRPO | per-stratum advantage normalisation |
| MS-GRPO | multi-scale advantage |
| Info-GRPO | mutual-information regulariser |
| Lambda-GRPO | learnable f_λ |
| AMIR-GRPO, DRA-GRPO, GAPO, GDPO, KRPO, PVPO, SPO, X-RPO | see [`configs/papers/`](configs/papers) |

A few methods that need trained reward models or multi-reward datasets are
intentionally not yet wired up — they would require infrastructure beyond
GSM8K-style single-signal training.

## Datasets

Ten datasets are wired in [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
with matching overlays in [`configs/data/`](configs/data). Pick one with
`--dataset-preset <name>` or include the overlay in your recipe.

| Preset | Source | `data_source` | veRL scorer | Status |
|---|---|---|---|---|
| `gsm8k` | `openai/gsm8k` | `openai/gsm8k` | `gsm8k` | ✅ runnable |
| `math` | `lighteval/MATH-Hard` | `lighteval/MATH` | `math_reward` | ✅ runnable |
| `dapo_math_17k` | `BytedTsinghua-SIA/DAPO-Math-17k` | `math_dapo` | `math_dapo` | ✅ runnable |
| `aime` | `Maxwell-Jia/AIME_2024` | `aime_2024` | `math_dapo` (via `aime*` prefix) | ✅ runnable |
| `amc` | `AI-MO/aimo-validation-amc` | `numina_amc_aime` | `prime_math` | ✅ runnable |
| `code_contests` | `deepmind/code_contests` | `codecontests` | `prime_code` / `sandbox_fusion` | ✅ runnable (needs sandbox URL or `prime_code`) |
| `gpqa` | `Idavidrein/gpqa` | `Idavidrein/gpqa` | none | ⚠️ needs custom scorer |
| `humaneval` | `openai_humaneval` | `openai_humaneval` | none | ⚠️ needs custom scorer |
| `mbpp` | `mbpp/sanitized` | `mbpp` | none | ⚠️ needs custom scorer |
| `livecodebench` | `livecodebench/code_generation_lite` | `livecodebench/code_generation_lite` | none | ⚠️ needs custom scorer |

⚠️ = preprocess writes parquets, but veRL 0.6.1's
[`default_compute_score`](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/reward_score/__init__.py)
will raise on the `data_source`. Either map to a supported scorer or wire
`reward_model.custom_reward_function` in your YAML. See
[`configs/data/README.md`](configs/data/README.md) for the full scorer
registry and the per-recipe compatibility matrix.

Switch dataset three equivalent ways:

```powershell
# CLI flag
python scripts/train_local.py --config examples/kalman_dapo.yaml --dataset-preset aime

# YAML data.preset (read by the launcher)
# examples/my_recipe.yaml: data: { preset: gpqa }
python scripts/train_local.py --config examples/my_recipe.yaml

# Direct parquet paths (bypass presets)
python scripts/train_local.py --config examples/kalman_dapo.yaml \
    --train-files /path/to/train.parquet --val-files /path/to/val.parquet
```

## Composer cookbook

The full menu of legal values per axis lives in
[`configs/custom_mix.yaml`](configs/custom_mix.yaml). Six worked recipes
that combine 2–4 components for specific failure modes live in
[`examples/`](examples) — see its [README](examples/README.md) for the
hypothesis / mechanism / risk on each one.

Rules of thumb when designing your own:

* `rollout.n >= 2` for any group method; `>= 4` for rank- or unlikeliness-
  based rewards.
* `trainer.balance_batch: false` always — group structure breaks otherwise
  (the trainer enforces this for you).
* `regularizer: none` removes the KL anchor; pair with a smaller clip range.
* `agg_mode: global_token` is length-fair; `token_mean` over-weights short
  responses.
* Stack stabilisers from **different axes** (baseline + clip + KL), not two
  on the same axis.

## Architecture (1-screen tour)

The **composer** lives in [`grpo_composer/core/`](grpo_composer/core). Each
subfolder is one axis of the gradient pipeline:

* `advantages/` — `standard`, `kalman`, `difficulty_aware`, `stratified`,
  `length_corrected`, `multi_scale`, `decoupled`, `pvpo`, `unbiased`, …
* `clipping/` — `symmetric`, `asymmetric`, `trajectory_level`,
  `weighted_trust`
* `aggregation/` — `token_mean`, `token_sum`, `global_token`,
  `group_uniform`, `weighted_token`, `group_learnable`, `difficulty_weighted`,
  `trajectory_level`
* `regularizers/` — `kl_divergence`, `log_weight`, `mutual_information`,
  `preference`
* `rewards/` — `binary`, `unlikeliness`, `rank_enhanced`, `rts_based`,
  `posterior_composite`, `length_dependent`, `multi_reward`,
  `diversity_adjusted`, `frequency_aware`

The **veRL integration** lives in
[`grpo_composer/integrations/verl/`](grpo_composer/integrations/verl).
Three composer subclasses own all of veRL's monkey-patch territory:

* `ComposerDataParallelPPOActor` — `update_policy`, `_forward_micro_batch`,
  `compute_log_prob` (surfaces hidden states for DRA-GRPO)
* `ComposerActorRolloutRefWorker` — swaps in the composer actor at
  `init_model` time
* `ComposerRayPPOTrainer` — composer `fit` loop with FlowPlugins
  (Info-GRPO, reference-reward), DAPO oversampling, Tracking/CSV fallback,
  forced `balance_batch: false`

The launcher
[`ComposerTaskRunner`](grpo_composer/integrations/verl/entrypoint.py)
registers the composer worker before `ray.remote(...)` wrapping and
instantiates the composer trainer. `scripts/train_grpo.py` redirects
`verl.trainer.main_ppo.run_ppo` to that entrypoint — the only launch-time
seam.

## Debugging

Three independent env switches (or YAML equivalents — Ray actors don't
inherit env vars, so prefer the YAML for cluster runs):

| Env var | YAML | Effect |
|---|---|---|
| `GRPO_COMPOSER_DEBUG=1` | — | Verbose composer init + per-step shape prints |
| `GRPO_COMPOSER_DAPO_DEBUG=1` | `algorithm.filter_groups.debug: true` | DAPO dynamic-sampling diagnostics |
| `GRPO_COMPOSER_STRICT_VALIDATION=1` | — | Raise on shape / config drift instead of warning |

`--debug` on the Modal/local launchers turns all three on.

## Docs

* [`docs/getting_started.md`](docs/getting_started.md) — install, first run, smoke test.
* [`docs/repository_state_and_experiment_guide.md`](docs/repository_state_and_experiment_guide.md) — paper-by-paper status, what's verified, what's flagged.
* [`examples/README.md`](examples/README.md) — mix-and-match cookbook recipes.

## License

See [LICENSE](LICENSE).
# Unified GRPO Framework v4

## Overview
This repository contains a unified mathematical framework for six Group Relative Policy Optimization (GRPO) variants. The framework consolidates these methods into a single objective function with configurable hyperparameters, allowing for the recovery of individual methods as well as the creation of hybrid configurations.

## Supported Methods
The framework unifies the following variants:
1.  **GRPO** (DeepSeekMath, 2024)
2.  **Dr. GRPO** (Bias-Free, 2025)
3.  **DAPO** (ByteDance, 2025)
4.  **DARO** (Difficulty-Aware, 2025)
5.  **$\lambda$-GRPO** (Token Preferences, 2025)
6.  **DRA-GRPO** (Diversity-Aware, 2025)

## Unified Objective
The master objective function is defined as:

$$
\mathcal{J}_{\text{Unified}}(\theta, \{w_\mu\}, \lambda) = \sum_{\mu \in \mathcal{M}} w_\mu^{\text{eff}} \cdot \mathbb{E}_{q: \mu_q = \mu} \left[ \mathbb{I}_{\text{OS}}(q) \cdot \frac{1}{\Omega_\mu} \sum_{i=1}^{G} f_\lambda(o_i) \cdot w_i^{\text{len}} \sum_{t=1}^{|o_i|} \left( \mathcal{L}_{\text{clip}}^{(i,t)} - \beta \cdot D_{\text{KL}}^{(i,t)} \right) \right] + \mathcal{L}_{\text{reg}}
$$

## Configuration
The framework is controlled by 19 hyperparameters, including:
-   **Clipping**: `epsilon_low`, `epsilon_high`
-   **Normalization**: `std_normalize`, `length_norm`, `group_norm_type`
-   **Regularization**: `beta` (KL penalty)
-   **Sampling**: `oversampling`, `group_size`
-   **Weighting**: `difficulty_weighting`, `lambda_weighting`, `diversity_weighting`

Refer to the mathematical documentation for detailed derivations and recovery proofs.
