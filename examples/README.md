# Composer Mix-and-Match Recipes

These YAMLs are **not** paper reproductions — they are designed combinations
that mix 2–4 composer components from different axes (advantage × clipping ×
aggregation × regularizer × reward pipeline) to explore unexplored corners
of the design space.

Every recipe runs on **GSM8K** out of the box: no extra reward model, no
extra dataset, no auxiliary signals. Just `--config examples/<name>.yaml`.

## Why a separate folder from `configs/`?

| Folder | Purpose | Stability |
|---|---|---|
| [`configs/`](../configs) | Faithful re-implementations from papers + base GRPO | **Pinned** — change only with citation |
| [`configs/papers/`](../configs/papers) | One file per published method (DAPO, KRPO, λ-GRPO, …) | **Pinned** to the paper's hyper-params |
| [`examples/`](.) | Hand-designed novel mixes meant for experimentation | **Mutable** — edit freely |

`configs/` is a fine name for the "spec" tree: it is what veRL/Hydra loads,
it maps 1-to-1 to published methods, and downstream tooling (CI, docs,
ablation scripts) treats those paths as stable. Cookbook recipes live
outside that contract on purpose.

## Index

| Recipe | Bottleneck attacked | Mix |
|---|---|---|
| [`kalman_dapo.yaml`](kalman_dapo.yaml) | Noisy baseline + clipped exploration + length penalty | KRPO Kalman advantage + asymmetric clip + global-token agg |
| [`diverse_explorer.yaml`](diverse_explorer.yaml) | Distribution sharpening (low pass\@N) | Unlikeliness reward + asymmetric clip (high=0.30) + no KL + n=8 |
| [`lambda_lead.yaml`](lambda_lead.yaml) | Hard-problem signal + token weighting | Difficulty-aware advantage + length-dep reward + λ-GRPO learnable agg |
| [`stable_explorer.yaml`](stable_explorer.yaml) | Long-run instability | KRPO Kalman + trajectory clip + weighted-KL |
| [`stratified_dapo.yaml`](stratified_dapo.yaml) | Heterogeneous batches | Stratified advantage + DAPO dynamic sampling + global-token |
| [`rank_sharp_grpo.yaml`](rank_sharp_grpo.yaml) | Sparse 0/1 reward signal | Rank-enhanced reward + asymmetric clip + group-uniform agg |

## How to launch

Modal:

```powershell
modal run scripts/train_modal.py --config examples/kalman_dapo.yaml
```

Local:

```powershell
python scripts/train_local.py --config examples/kalman_dapo.yaml
```

Both launchers resolve the path relative to the repo root, so the leading
`examples/` is enough.

## How each file is structured

Three comment blocks at the top of every recipe:

* **HYPOTHESIS** — what failure mode of vanilla GRPO this combo targets.
* **WHY IT MIGHT WIN** — the mechanism that makes the mix more than the sum
  of its parts.
* **RISK** — what to watch for and which knob to back off first.

Below the comments, the YAML follows the same skeleton as the paper configs:

```yaml
algorithm:
  adv_estimator: <one of the composer estimators>
  # plus per-estimator hyperparams
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: composer
  rollout:
    n: 4
composer:
  clip_mode: ...
  agg_mode: ...
  regularizer: ...
  composer_reward_pipeline: [ ... ]   # optional
trainer:
  balance_batch: false                # ALWAYS off for any group method
```

## Building your own

The cookbook in [`../configs/custom_mix.yaml`](../configs/custom_mix.yaml)
lists every legal value for each axis. Rules of thumb:

* `rollout.n >= 2` for any group method (GRPO, DAPO, KRPO, λ-GRPO, …); `>= 4`
  for rank-based or unlikeliness rewards.
* `trainer.balance_batch: false` — group structure breaks otherwise.
* `regularizer: none` removes the KL anchor — pair with a **smaller** clip
  range to keep updates bounded, or back off to `kl` with `reg_coef: 0.005`.
* `agg_mode: global_token` is fairer when responses have very different
  lengths; `token_mean` over-weights short responses.
* Reward pipelines compose left-to-right; keep them short (1–2 entries).
* Stack stabilisers from **different axes** (baseline + clip + KL), not two
  on the same axis — they tend to cancel rather than compose.
# Composer Mix-and-Match Recipes

These YAMLs are **not** paper reproductions — they are designed combinations
that mix 2–4 composer components from different axes to explore unexplored
corners of the design space. Every recipe is runnable on GSM8K out of the box
(no extra reward model, no extra dataset).

Pick by the *bottleneck* you want to attack:

| Recipe | Bottleneck attacked | Mix |
|---|---|---|
| [`kalman_dapo.yaml`](kalman_dapo.yaml) | Noisy baseline + clipped exploration | Kalman advantage + asymmetric clip + global-token agg |
| [`diverse_explorer.yaml`](diverse_explorer.yaml) | Distribution sharpening (low pass\@N) | Unlikeliness reward + asymmetric clip + no KL |
| [`lambda_lead.yaml`](lambda_lead.yaml) | Hard-problem signal + token weighting | Difficulty-aware adv + length-dep reward + λ-GRPO learnable aggregation |
| [`stable_explorer.yaml`](stable_explorer.yaml) | Long-run instability | KRPO Kalman + trajectory clip + weighted-KL |
| [`stratified_dapo.yaml`](stratified_dapo.yaml) | Heterogeneous batches | Stratified advantage + DAPO dynamic sampling + global-token agg |
| [`rank_sharp_grpo.yaml`](rank_sharp_grpo.yaml) | Sparse 0/1 reward signal | Rank-enhanced reward + asymmetric clip + group-uniform agg |

## How to launch one

```powershell
modal run scripts/train_modal.py --config configs/examples/kalman_dapo.yaml
```

or locally:

```powershell
python scripts/train_local.py --config configs/examples/kalman_dapo.yaml
```

## Reading a recipe

Each file has the same skeleton:

```yaml
algorithm:
  adv_estimator: <one of the composer estimators>
  # plus per-estimator hyperparams
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: composer
  rollout:
    n: 4
composer:
  clip_mode: ...
  agg_mode: ...
  regularizer: ...
  composer_reward_pipeline: [ ... ]   # optional
trainer:
  balance_batch: false                # ALWAYS off for any group method
```

Header comments in each file explain the **hypothesis** (why this combo
might beat vanilla GRPO) and the **risk** (what could go wrong).

## Mixing your own

The cookbook in [`../custom_mix.yaml`](../custom_mix.yaml) lists every legal
value for each axis. Rules of thumb:

* `rollout.n >= 2` for any group method (GRPO, DAPO, KRPO, λ-GRPO, …).
* `trainer.balance_batch: false` — group structure breaks otherwise.
* `regularizer: none` removes the KL anchor — pair with **smaller** clip range
  (e.g. `clip_ratio: 0.2`) to keep updates bounded.
* `agg_mode: global_token` is fairer when responses have very different
  lengths; `token_mean` over-weights short responses.
* Reward pipelines compose left-to-right; keep them short (1–2 entries).
