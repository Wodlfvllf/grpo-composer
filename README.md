# grpo-composer

> **A Unified, Component-Driven Library for Critic-Free Reinforcement Learning in Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```bash
pip install grpo-composer
```

```python
from grpo_composer import GRPOTrainer
from grpo_composer.rewards import FrequencyAwareReward
from grpo_composer.advantages import KalmanAdvantage

trainer = GRPOTrainer(
    model=model,
    reward_engine=FrequencyAwareReward(),  # GAPO
    advantage_estimator=KalmanAdvantage(), # KRPO
)
trainer.train(dataset)
```

## 📚 Supported Papers (22)

KRPO, GAPO, Dr.GRPO, DRA-GRPO, DAPO, DARO, λ-GRPO, GDPO, GRPO-LEAD, MS-GRPO, P-GRPO, PVPO, RankGRPO, Unlikeliness-GRPO, SPO, Stratified-GRPO, TIC-GRPO, TR-GRPO, XRPO, AMIR-GRPO, Info-GRPO

## 📄 License

MIT License
