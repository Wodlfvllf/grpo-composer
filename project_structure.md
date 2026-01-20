# grpo-composer

> **A Unified, Component-Driven Library for Critic-Free Reinforcement Learning in Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

**grpo-composer** is the first comprehensive, modular library that unifies 22+ GRPO variants into a single framework. By deconstructing the GRPO algorithm into atomic, interchangeable components, researchers can "mix and match" state-of-the-art techniques via simple configuration.

---

## 📁 Repository Structure

```
grpo-composer/
│
├── 📄 README.md                      # This file
├── 📄 pyproject.toml                 # Package configuration
├── 📄 LICENSE
├── 📄 CONTRIBUTING.md
│
├── 📂 configs/                       # YAML configurations
│   ├── base_grpo.yaml
│   ├── 📂 papers/                    # Paper reproduction configs
│   │   ├── krpo.yaml
│   │   ├── gapo.yaml
│   │   ├── dr_grpo.yaml
│   │   ├── dapo.yaml
│   │   ├── daro.yaml
│   │   ├── lambda_grpo.yaml
│   │   ├── dra_grpo.yaml
│   │   ├── gdpo.yaml
│   │   ├── grpo_lead.yaml
│   │   ├── ms_grpo.yaml
│   │   ├── p_grpo.yaml
│   │   ├── pvpo.yaml
│   │   ├── rank_grpo.yaml
│   │   ├── unlikeliness_grpo.yaml
│   │   ├── spo.yaml
│   │   ├── stratified_grpo.yaml
│   │   ├── tic_grpo.yaml
│   │   ├── tr_grpo.yaml
│   │   ├── xrpo.yaml
│   │   ├── amir_grpo.yaml
│   │   └── info_grpo.yaml
│   └── 📂 experiments/
│       ├── math_reasoning.yaml
│       ├── code_generation.yaml
│       └── agentic_search.yaml
│
├── 📂 grpo_composer/                 # 🔥 Main Package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── 📂 core/                      # Core abstractions
│   │   ├── base.py                   # BaseComponent protocols
│   │   ├── registry.py               # Component registry
│   │   ├── config.py                 # Config management
│   │   └── trainer.py                # GRPOTrainer
│   │
│   ├── 📂 rewards/                   # 🎯 Reward Engines (10 modules)
│   │   ├── base.py
│   │   ├── binary.py                 # Standard binary
│   │   ├── frequency_aware.py        # GAPO
│   │   ├── diversity_adjusted.py     # DRA-GRPO
│   │   ├── length_dependent.py       # GRPO-LEAD
│   │   ├── posterior_composite.py    # P-GRPO
│   │   ├── rank_enhanced.py          # RankGRPO
│   │   ├── unlikeliness.py           # Unlikeliness-GRPO
│   │   ├── rts_based.py              # SPO
│   │   └── multi_reward.py           # GDPO
│   │
│   ├── 📂 advantages/                # 📊 Advantage Estimators (12 modules)
│   │   ├── base.py
│   │   ├── standard.py               # (r - μ) / σ
│   │   ├── unbiased.py               # Dr.GRPO
│   │   ├── kalman.py                 # KRPO
│   │   ├── static_value.py           # PVPO
│   │   ├── decoupled.py              # GDPO
│   │   ├── multi_scale.py            # MS-GRPO
│   │   ├── difficulty_aware.py       # GRPO-LEAD
│   │   ├── length_corrected.py       # TIC-GRPO
│   │   ├── stratified.py             # Stratified-GRPO
│   │   ├── advantage_clipping.py     # RankGRPO
│   │   └── novelty_sharpening.py     # XRPO
│   │
│   ├── 📂 clipping/                  # ✂️ Clipping Mechanisms (5 modules)
│   │   ├── base.py
│   │   ├── symmetric.py              # Standard
│   │   ├── asymmetric.py             # DAPO
│   │   ├── trajectory_level.py       # TIC-GRPO
│   │   └── weighted_trust.py         # TR-GRPO
│   │
│   ├── 📂 regularizers/              # 🔗 Regularizers (6 modules)
│   │   ├── base.py
│   │   ├── kl_divergence.py          # Standard KL
│   │   ├── weighted_kl.py            # TR-GRPO
│   │   ├── preference.py             # AMIR-GRPO
│   │   ├── difficulty_balance.py     # DARO
│   │   └── info_regularizer.py       # Info-GRPO
│   │
│   ├── 📂 aggregation/               # ⚖️ Token/Group Aggregation (9 modules)
│   │   ├── base.py
│   │   ├── token_mean.py             # 1/|o_i|
│   │   ├── token_sum.py              # Dr.GRPO
│   │   ├── global_token.py           # DAPO
│   │   ├── trajectory_level.py       # TIC-GRPO
│   │   ├── weighted_token.py         # TR-GRPO
│   │   ├── group_uniform.py          # 1/G
│   │   ├── group_learnable.py        # λ-GRPO
│   │   └── difficulty_weighted.py    # DARO
│   │
│   ├── 📂 sampling/                  # 🎲 Sampling Strategies (6 modules)
│   │   ├── base.py
│   │   ├── standard.py               # Uniform
│   │   ├── dynamic.py                # DAPO
│   │   ├── difficulty_grouped.py     # DARO
│   │   ├── gt_injection.py           # PVPO
│   │   └── hierarchical.py           # XRPO
│   │
│   ├── 📂 models/                    # Model utilities
│   │   ├── policy.py
│   │   ├── reference.py
│   │   ├── reward_model.py
│   │   └── embeddings.py
│   │
│   ├── 📂 losses/
│   │   ├── base.py
│   │   └── grpo_loss.py              # Unified loss
│   │
│   └── 📂 utils/
│       ├── logging.py
│       ├── metrics.py
│       ├── checkpointing.py
│       ├── distributed.py
│       └── math_utils.py
│
├── 📂 scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── reproduce_paper.py
│   └── ablation.py
│
├── 📂 examples/
│   ├── quickstart.ipynb
│   ├── custom_component.py
│   └── mix_and_match.py
│
├── 📂 tests/
│   ├── test_rewards/
│   ├── test_advantages/
│   ├── test_clipping/
│   ├── test_regularizers/
│   ├── test_aggregation/
│   ├── test_sampling/
│   ├── test_integration/
│   └── test_paper_recovery/          # Exact paper reproduction tests
│
├── 📂 benchmarks/
│   ├── math_reasoning/
│   ├── code_generation/
│   └── memory_profiling/
│
└── 📂 docs/
    ├── index.md
    ├── getting_started.md
    ├── 📂 concepts/
    ├── 📂 papers/
    ├── 📂 api/
    └── 📂 tutorials/
```

---

## 🗺️ Component Mapping (22 Papers)

| Paper | Reward | Advantage | Clipping | Regularizer | Aggregation | Sampling |
|:------|:------:|:---------:|:--------:|:-----------:|:-----------:|:--------:|
| **GRPO** | binary | standard | symmetric | kl | token_mean | standard |
| **KRPO** | - | kalman | - | - | - | - |
| **GAPO** | frequency | - | - | - | - | - |
| **Dr.GRPO** | - | unbiased | - | none | token_sum | - |
| **DRA-GRPO** | diversity | - | - | - | - | - |
| **DAPO** | - | - | asymmetric | none | global_token | dynamic |
| **DARO** | - | - | - | difficulty | difficulty_wt | difficulty |
| **λ-GRPO** | - | - | - | - | learnable | - |
| **GDPO** | multi | decoupled | - | - | - | - |
| **GRPO-LEAD** | length | difficulty | - | none | - | - |
| **MS-GRPO** | - | multi_scale | - | - | - | - |
| **P-GRPO** | composite | - | asymmetric | none | - | - |
| **PVPO** | - | static_v | - | - | - | gt_inject |
| **RankGRPO** | rank | adv_clip | - | - | - | - |
| **Unlikeliness** | unlikely | - | - | - | - | - |
| **SPO** | rts | - | - | - | - | - |
| **Stratified** | - | stratified | - | - | - | - |
| **TIC-GRPO** | - | length_corr | trajectory | - | trajectory | - |
| **TR-GRPO** | - | - | weighted | weighted_kl | weighted | - |
| **XRPO** | - | novelty | - | - | - | hierarchical |
| **AMIR-GRPO** | - | - | - | preference | - | - |
| **Info-GRPO** | - | - | - | info | - | - |

---

## 🚀 Quick Start

### Installation
```bash
pip install grpo-composer
```

### Basic Usage
```python
from grpo_composer import GRPOTrainer
from grpo_composer.rewards import FrequencyAwareReward
from grpo_composer.advantages import KalmanAdvantage

# Mix GAPO's reward with KRPO's advantage
trainer = GRPOTrainer(
    model=model,
    reward_engine=FrequencyAwareReward(),       # GAPO
    advantage_estimator=KalmanAdvantage(),      # KRPO
)
trainer.train(dataset)
```

### Via Config
```yaml
reward_engine: frequency_aware      # GAPO
advantage_estimator: kalman         # KRPO
clipping: asymmetric                # DAPO
regularizer: preference             # AMIR-GRPO
```

```bash
python scripts/train.py --config configs/custom.yaml
```

### Reproduce a Paper
```bash
python scripts/reproduce_paper.py --paper krpo
```

---

## 📚 Supported Papers (22)

1. **GRPO** - DeepSeekMath (2024)
2. **KRPO** - Kalman Filter Posterior
3. **GAPO** - Group-Aware Frequency Rewards
4. **Dr.GRPO** - Bias-Free Gradients
5. **DRA-GRPO** - Diversity via SMI
6. **DAPO** - Asymmetric Clipping
7. **DARO** - Difficulty-Aware Weighting
8. **λ-GRPO** - Learnable Length Weights
9. **GDPO** - Multi-Reward Decoupling
10. **GRPO-LEAD** - Length + Difficulty
11. **MS-GRPO** - Multi-Scale Advantages
12. **P-GRPO** - Posterior Thinking Reward
13. **PVPO** - Static Value Baseline
14. **RankGRPO** - Ranking as Reward
15. **Unlikeliness-GRPO** - Rare Solution Boost
16. **SPO** - Reasoning Trajectory Score
17. **Stratified-GRPO** - Per-Stratum Normalization
18. **TIC-GRPO** - Trajectory-Level Importance
19. **TR-GRPO** - Token-Regulated Sharpness
20. **XRPO** - Exploration-Exploitation Planning
21. **AMIR-GRPO** - DPO-Style Preference
22. **Info-GRPO** - Mutual Information Regularizer

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
