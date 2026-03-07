# GRPO Composer Repository State and Experiment Guide

Last updated: March 7, 2026
Audience: New contributors who need a complete map of what is already built, what is partially built, and what is still missing before heavy patching and experimentation.

## 1. Executive Summary

This repository has a strong core implementation of composable GRPO components and a working veRL integration path, with a verified Modal smoke run for base GRPO.

At the same time, several parts of the repository are still scaffolding:
- docs are mostly empty
- examples are empty
- recipe files are empty
- the `grpo_composer/config` package is empty
- test coverage is narrow relative to implemented features

The practical state is:
- Core math components: mostly implemented
- veRL integration: implemented and actively used
- Modal training path: implemented and recently exercised
- Paper reproducibility completeness: mixed (some paper configs are runnable now, others need additional batch signals or architecture work)
- Documentation quality: not yet aligned with code complexity

## 2. What Was Audited

This guide was produced after reading:
- user walkthrough file: `/Users/shashank/Deep_Learning/guide of what has been done Walkthrough`
- repository root and source tree under `/Users/shashank/Deep_Learning/GRPO Library/grpo_composer`

Key files inspected include:
- `README.md`
- `pyproject.toml`
- all files in `configs/`
- all files in `scripts/`
- all files in `grpo_composer/integrations/verl/`
- representative core files under `grpo_composer/core/`
- all Python tests under `tests/`
- docs, recipes, and examples directories

## 3. Repository Layout and Current Completion Level

### 3.1 Top-level structure

- `grpo_composer/core/`: primary algorithmic implementation (implemented)
- `grpo_composer/integrations/verl/`: veRL adapter layer and trainer patching (implemented)
- `configs/`: base + paper configs (implemented, but some paper fidelity gaps)
- `scripts/`: training and patch scripts (partially modernized)
- `tests/`: a small subset of implemented behavior is tested
- `docs/`, `examples/`, `recipes/`, `configs/experiments/`: currently placeholders (mostly empty)

### 3.2 Empty file hotspots (important)

The following areas are mostly empty and should be treated as missing documentation/usability layers rather than complete features:

- `docs/` (pre-existing markdown pages are empty placeholders; this guide was added during the audit)
- `examples/` (all example `.py` files empty)
- `recipes/` (all recipe files empty)
- `configs/experiments/` (all experiment yaml files empty)
- `grpo_composer/config/` (all config helper modules empty)
- `grpo_composer/core/base.py` empty
- `grpo_composer/version.py` empty
- `grpo_composer/integrations/__init__.py` empty
- `grpo_composer/core/clipping/__init__.py` empty

## 4. Implemented Core Library Components

The mathematical core is substantial and not just stubs.

### 4.1 Advantages implemented

Classes present under `grpo_composer/core/advantages/`:
- `StandardAdvantageFunction`
- `DecoupledAdvantageFunction`
- `DifficultyAwareAdvantageFunction`
- `LengthCorrectedAdvantageFunction`
- `KalmanAdvantageFunction`
- `MultiScaleAdvantageFunction`
- `NoveltySharpeningAdvantageFunction`
- `StaticValueAdvantageFunction`
- `StratifiedAdvantageFunction`
- `UnbiasedAdvantageFunction`
- `AdvantageClipping`

### 4.2 Clipping mechanisms implemented

Under `grpo_composer/core/clipping/`:
- `SymmetricClippingMechanism`
- `AsymmetricClippingMechanism`
- `TrajectoryLevelClippingMechanism`
- `WeightedTrustRegionClippingMechanism`

### 4.3 Aggregation methods implemented

Under `grpo_composer/core/aggregation/`:
- `TokenMeanAggregation`
- `TokenSumAggregation`
- `GlobalTokenAggregation`
- `GroupUniformAggregation`
- `TrajectoryLevelAggregation`
- `DifficultyWeightedAggregation`
- `GroupLearnableAggregation`
- `WeightedTokenAggregation`

### 4.4 Regularizers implemented

Under `grpo_composer/core/regularizers/`:
- `KLDivergenceRegularizer`
- `WeightedKLDivergenceRegularizer`
- `PreferenceRegularizer`
- `LogWeightRegularizer`
- `MutualInformationRegularizer`

### 4.5 Reward calculators implemented

Under `grpo_composer/core/rewards/`:
- `BinaryRewardCalculator`
- `DiversityAdjustedRewardCalculator`
- `FrequencyAwareRewardCalculator`
- `LengthDependentRewardCalculator`
- `PosteriorCompositeRewardCalculator`
- `MultiRewardProcessor`
- `RankEnhancedRewardCalculator`
- `RTSRewardCalculator`
- `UnlikelinessRewardCalculator`

### 4.6 Legacy/alternate loss abstraction

`grpo_composer/core/losses/` contains `LossFunction` and `CustomLoss` abstractions, but these are not what the active veRL path uses in production runs. Current runs use the registered `composer` policy loss in `grpo_composer/integrations/verl/losses.py`.

## 5. veRL Integration Status

This is the most important working integration layer in the repo.

### 5.1 Registries wired into veRL

From `grpo_composer/integrations/verl/`:
- policy loss registration:
  - `@register_policy_loss("composer")`
- advantage estimator registrations:
  - `difficulty_aware_grpo`
  - `length_corrected_grpo`
  - `kalman_grpo`
  - `decoupled_grpo`
  - `multi_scale_grpo`
  - `static_value_grpo`
  - `novelty_sharp_grpo`
  - `stratified_grpo`
  - `unbiased_grpo`

### 5.2 Custom trainer patching

`patch_verl_main_ppo()` in `grpo_composer/integrations/verl/trainer.py` patches:
- `RayPPOTrainer -> ComposerRayPPOTrainer`
- `compute_advantage -> composer_compute_advantage`

### 5.3 Reward transform pipeline in trainer

Implemented transforms:
- `_apply_unlikeliness_reward_transform`
- `_apply_rank_enhanced_reward_transform`
- `_apply_rts_reward_transform`
- `_apply_posterior_reward_transform`
- `_apply_multi_reward_transform`
- `_apply_length_dependent_reward_transform`

### 5.4 Flow plugin system

Implemented flow plugins:
- `default`
- `info_grpo`
- `pvpo`

### 5.5 Key runtime guardrails in trainer

The trainer enforces constraints that directly affect config readiness:
- `group_learnable + lambda_learnable=true` is explicitly blocked in `_validate_supported_modes`
- multiple estimators and reward transforms raise clear errors when required batch fields are missing

## 6. Training and Runtime Infrastructure

### 6.1 Current active Modal path

`scripts/train_modal.py` is currently the most complete entrypoint.

Current pinned stack in this script:
- `verl==0.6.1`
- `vllm==0.8.5`
- `ray[default]>=2.40.0`
- `transformers>=4.51.0,<5.0.0`
- `tensordict>=0.8.0,<=0.10.0,!=0.9.0`

This aligns with your latest stabilization effort around vLLM engine failures.

### 6.2 What `train_modal.py` already auto-injects

Notable defaults and safety overrides include:
- dataset auto-prep for GSM8K
- actor/critic/ref/rollout micro-batch defaults
- `rollout.name=vllm`
- rollout memory/stability knobs (`max_model_len`, `gpu_memory_utilization`, eager, caching toggles)
- attention override to `sdpa`
- worker-side registration via:
  - `++actor_rollout_ref.model.external_lib=grpo_composer.integrations.verl`

This last point is critical for avoiding `Unsupported loss mode: composer` in worker processes.

### 6.3 Other scripts

- `scripts/train_grpo.py`: active veRL launcher with patching
- `scripts/train.sh`: present but appears stale relative to current config style and data keys
- `scripts/train_multinode.sh`: present but likely stale relative to current hydra override conventions
- `scripts/patch_verl_collective_rpc.py`: compatibility patch for vLLM 0.11.2 async interface drift
- `scripts/patch_verl_loss_registry.py`: fallback patch for lazy composer loss registration in veRL workers

Note: patch scripts are currently optional utilities and are not automatically executed by `train_modal.py`.

## 7. Verified Runtime Progress

From your latest logs:
- Base GRPO smoke run completed to 100% (`7/7`)
- checkpoints written under `/checkpoints/grpo_smoke_test`
- actor update, generation, and checkpointing all executed end-to-end

This confirms the training pipeline is operational for at least the base path.

## 8. Config Inventory and Readiness Assessment

Below is a practical readiness table based on actual config values plus hard requirements in integration code.

Legend:
- Ready now: expected to run with current pipeline and available batch fields
- Data-dependent: code exists but extra batch columns/signals must be provided
- Architecture work needed: requires rollout/control-flow changes not yet present
- Paper-fidelity mismatch: config comments describe behavior not currently activated

### 8.1 Ready now (or close to ready)

- `configs/base_grpo.yaml`
- `configs/papers/dapo.yaml`
- `configs/papers/dr_grpo.yaml`
- `configs/papers/krpo.yaml`
- `configs/papers/ms_grpo.yaml` (compute heavy, but implemented)
- `configs/papers/tic_grpo.yaml`
- `configs/papers/tr_grpo.yaml`
- `configs/papers/xrpo.yaml`
- `configs/papers/amir_grpo.yaml`
- `configs/papers/grpo_lead.yaml` (uses built length-dependent reward transform)
- `configs/papers/rank_grpo.yaml` (if `old_log_probs` are present, as expected)
- `configs/papers/rewarding_unlikely.yaml` (same note on `old_log_probs`)

### 8.2 Data-dependent (extra fields/signals required)

- `configs/papers/gdpo.yaml`
  - requires multi-reward tensors for decoupled advantage and/or multi-reward transform
- `configs/papers/pvpo.yaml`
  - `static_value_grpo` requires reference rewards in batch
- `configs/papers/stratified_grpo.yaml`
  - requires `strata`/`stratum_ids`
- `configs/papers/spo.yaml`
  - requires RTS scores (external scorer integration)
- `configs/papers/p_grpo.yaml`
  - posterior pipeline requires `format/outcome/thinking` reward channels
- `configs/papers/posterior_grpo.yaml`
  - same posterior reward requirements

### 8.3 Architecture work needed

- `configs/papers/info_grpo.yaml`
  - mutual info regularizer requires augmented rollout tensors (`log_probs_aug`, `mask_aug`)
  - this typically needs dual rollout logic or equivalent augmentation flow not currently wired end-to-end

### 8.4 Currently blocked by explicit trainer constraint

- `configs/papers/lambda_grpo.yaml`
  - sets `lambda_learnable: true`
  - current trainer explicitly rejects `group_learnable + lambda_learnable=true`
  - needs optimizer/plumbing work before faithful training

### 8.5 Paper-fidelity mismatches in config semantics

Important mismatches between comments and active knobs:

- `configs/papers/daro.yaml`
  - comments describe difficulty-weighted aggregation + log-weight regularizer
  - actual config uses `agg_mode: token_mean`, `regularizer: none`
  - result: runnable, but not faithful to described DARO behavior

- `configs/papers/gapo.yaml`
  - comments describe frequency-aware reward objective
  - config itself does not activate a frequency-aware reward transform in trainer pipeline
  - result: runnable, but mostly base-like unless reward manager injects this externally

- `configs/papers/dra_grpo.yaml`
  - comments describe diversity-adjusted reward usage
  - no corresponding reward transform is activated in current trainer pipeline
  - result: runnable, but not faithful without external reward shaping path

## 9. Test Coverage Reality

Current test coverage is limited compared to implementation scope.

### 9.1 What is tested

- `tests/test_rewards/test_rewards.py`
  - unit tests for reward calculators
- `tests/test_verl/test_trainer_pipeline.py`
  - tests for selected trainer reward transforms and context injection

### 9.2 What is not tested (major gaps)

No substantive tests currently exist for:
- clipping modules
- aggregation modules
- advantage modules (beyond indirect coverage)
- regularizer modules (beyond indirect behavior)
- per-paper config recovery tests
- integration-level end-to-end config sanity checks
- Modal launcher override correctness tests

Empty test directories indicate planned but unfinished coverage:
- `tests/test_advantages/`
- `tests/test_aggregation/`
- `tests/test_clipping/`
- `tests/test_regularizers/`
- `tests/test_paper_recovery/`

## 10. Documentation and Usability Gaps

### 10.1 Documentation

Before this guide was added, the pre-existing docs pages under `docs/` were empty placeholders. There is still no complete canonical onboarding path yet beyond sparse `README.md` plus this audit document.

### 10.2 Examples and recipes

Examples and recipe files are placeholders and currently do not provide runnable guidance.

### 10.3 Config helper package

`grpo_composer/config/` is empty, which means no central schema/validator/builder layer is currently available despite expected module names.

### 10.4 README drift

`README.md` still describes an earlier scope and does not reflect:
- expanded paper variants
- trainer patch architecture
- runtime dependency constraints
- real deployment path on Modal

## 11. Operational and Repository Hygiene Gaps

### 11.1 Build artifacts tracked in repo

There are many committed `.pyc` files and `__pycache__` directories in the working tree. This is noise and should be removed from version control history moving forward.

### 11.2 Script consistency

The main modern path is `scripts/train_modal.py`. Other wrappers appear less aligned with the current hydra override style and may cause confusion for newcomers.

### 11.3 Local changes state

Current git status indicates local modifications/untracked files in training-related scripts. This should be explicitly tracked in a changelog or PR notes before broader experiments.

## 12. Known Runtime Failure Modes Encountered Recently

Based on your run history, the following issues are known and relevant:

- `ModuleNotFoundError` for local integration imports in container workers
  - addressed through packaging/path adjustments

- `Unsupported loss mode: composer`
  - caused by missing worker-side registration
  - mitigated via `external_lib` import strategy

- Flash Attention import failures with Qwen and some dependency mixes
  - mitigated via attention override to `sdpa`

- vLLM EngineDeadError (`vllm/v1/engine/core.py` -> `future.result()` on `None`)
  - seen with newer vLLM/veRL combinations
  - current practical mitigation is stable version pinning (veRL 0.6.1 + vLLM 0.8.5)
  - optional patch scripts exist for newer interface drift, but these are not yet formalized into a tested release path

## 13. What Is Completed vs Missing (Concise Matrix)

### 13.1 Completed enough for immediate experiments

- composable core components for clipping/aggregation/regularization/advantages/rewards
- veRL integration and trainer patching
- base config and multiple paper-like configs
- Modal smoke path for base GRPO

### 13.2 Partially completed

- paper config set (many runnable, some not paper-faithful, some data-dependent)
- reward transform infrastructure (several transforms implemented, but not all reward calculators are wired into trainer pipeline)
- test suite (partial)

### 13.3 Not completed

- docs, tutorials, API references in repo docs folder
- examples and recipes
- config schema/validation package
- full paper reproduction pipeline requirements (for data-dependent and architecture-dependent variants)

## 14. Priority Roadmap Before Heavy Patching/Experiments

### P0 (do first)

1. Document canonical runtime stack and enforce it in one place.
2. Define one official launcher path (recommend Modal script + one local fallback) and mark others as legacy or update them.
3. Add end-to-end configuration sanity checks that fail early when required signals are missing.
4. Remove tracked `.pyc`/`__pycache__` artifacts from version control.
5. Replace empty docs with minimal operational docs (this file is first step).

### P1 (for reliable paper experimentation)

1. Complete data pipelines for variants requiring extra fields:
   - multi-reward tensors
   - reference rewards
   - strata ids
   - RTS scores
   - posterior reward channels
2. Implement or integrate dual-rollout path for Info-GRPO.
3. Add lambda optimizer/plumbing to enable `lambda_learnable=true` variants.
4. Add tests for each component family plus config-level readiness tests.

### P2 (developer experience)

1. Fill examples and recipes with runnable scripts.
2. Implement `grpo_composer/config/*` schema/validation/builder modules.
3. Refresh README to reflect real architecture and supported workflows.

## 15. Pre-Experiment Checklist for New Contributors

Use this checklist before launching a new variant:

1. Confirm dependency matrix (veRL, vLLM, ray, transformers, torch).
2. Confirm worker-side registration path is active (`external_lib` or equivalent).
3. Verify required inputs for the chosen `adv_estimator` and `composer_reward_pipeline`.
4. Run a 1-epoch smoke test with tiny batch size and checkpoint save enabled.
5. Inspect first-step metrics and ensure loss mode and estimator are as intended.
6. Save command + resolved overrides for reproducibility.
7. Promote to longer run only after smoke pass and checkpoint integrity check.

## 16. Suggested Immediate Next Work Items

If the goal is to start serious variant experiments quickly, the most leverage comes from:

1. Add a "variant readiness validator" script that inspects selected config and asserts required batch keys before run.
2. Implement one standardized dataset enrichment pipeline for:
   - `strata`
   - `multi_rewards`
   - `reference_rewards`
3. Decide whether to stay on the stable vLLM 0.8.5 line for now or formalize and test the vLLM 0.11+ patch path.
4. Add 3 high-value integration tests:
   - composer loss registration in worker context
   - rank/unlikeliness reward pipeline on synthetic batch
   - one smoke config per readiness category

## 17. Final Notes for Newcomers

This repository has real algorithmic depth and a functional training path, but its user-facing and reproducibility scaffolding lags behind implementation.

Treat it as:
- a strong research kernel with many implemented components
- plus an unfinished product layer (docs/examples/recipes/validation/test breadth)

For safe progress, prioritize consistency and validation plumbing first, then launch broader experiments.
