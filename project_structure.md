```markdown
grpo_composer/
│
├── grpo_composer/
│   ├── __init__.py
│   │
│   ├── core/                           # Core abstractions (NEVER modify)
│   │   ├── __init__.py
│   │   ├── protocols.py                # Protocol definitions (interfaces)
│   │   ├── registry.py                 # Global component registry
│   │   ├── pipeline.py                 # Pipeline executor
│   │   └── base.py                     # Base classes (ComposableOp, etc.)
│   │
│   ├── stages/                         # Pipeline stages (ordered)
│   │   ├── __init__.py
│   │   ├── 01_sampling.py              # Stage: Sample G outputs
│   │   ├── 02_reward.py                # Stage: Compute raw rewards
│   │   ├── 03_reward_shaping.py        # Stage: DRA, filtering, etc.
│   │   ├── 04_advantage.py             # Stage: Advantage computation
│   │   ├── 05_weighting.py             # Stage: Response/difficulty weights
│   │   ├── 06_loss.py                  # Stage: Clipped surrogate + KL
│   │   └── 07_aggregation.py           # Stage: Final aggregation
│   │
│   ├── components/                     # Pluggable implementations
│   │   ├── __init__.py
│   │   ├── advantage/
│   │   │   ├── __init__.py
│   │   │   ├── grpo.py                 # A = (R - μ) / σ
│   │   │   ├── dr_grpo.py              # A = R - μ (no σ)
│   │   │   └── base.py                 # AdvantageComputer protocol
│   │   ├── clipping/
│   │   │   ├── __init__.py
│   │   │   ├── symmetric.py            # clip(ρ, 1-ε, 1+ε)
│   │   │   ├── asymmetric.py           # clip(ρ, 1-ε_l, 1+ε_h)
│   │   │   └── base.py                 # Clipper protocol
│   │   ├── reward_shaping/
│   │   │   ├── __init__.py
│   │   │   ├── identity.py             # No modification
│   │   │   ├── diversity.py            # R̃ = R · (1 - SMI)
│   │   │   └── base.py                 # RewardShaper protocol
│   │   ├── weighting/
│   │   │   ├── __init__.py
│   │   │   ├── uniform.py              # w = 1
│   │   │   ├── lambda_weight.py        # f_λ(o_i)
│   │   │   ├── difficulty_weight.py    # w_μ (DARO)
│   │   │   └── base.py                 # Weighter protocol
│   │   ├── normalization/
│   │   │   ├── __init__.py
│   │   │   ├── per_response.py         # 1/|o_i|
│   │   │   ├── per_group.py            # 1/G
│   │   │   ├── per_token_total.py      # 1/Σ|o_i|
│   │   │   └── base.py                 # Normalizer protocol
│   │   └── filtering/
│   │       ├── __init__.py
│   │       ├── none.py                 # Include all
│   │       ├── oversampling.py         # I[0 < μ_q < 1]
│   │       └── base.py                 # Filter protocol
│   │
│   ├── objectives/                     # Pre-built objectives
│   │   ├── __init__.py
│   │   ├── unified.py                  # UnifiedObjective(nn.Module)
│   │   └── factory.py                  # from_config() factory
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py                   # Pydantic config schema
│   │   └── presets/                    # YAML presets
│   │       ├── grpo.yaml
│   │       ├── dr_grpo.yaml
│   │       ├── dapo.yaml
│   │       ├── daro.yaml
│   │       ├── lambda_grpo.yaml
│   │       └── dra_grpo.yaml
│   │
│   └── utils/
│       ├── __init__.py
│       └── tensor_ops.py
│
├── examples/
│   ├── 01_quick_start.py
│   ├── 02_custom_component.py          # How to add your own
│   └── 03_hybrid_objective.py
│
└── tests/
    ├── test_registry.py
    ├── test_pipeline.py
    └── test_recovery.py
```
