unified_grpo/
│
├── unified_grpo/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py              # 19 hyperparameter definitions
│   │   ├── presets.py             # GRPO, DAPO, DARO, etc. presets
│   │   └── validator.py           # Config validation
│   │
│   ├── operations/                 # Atomic building blocks
│   │   ├── __init__.py
│   │   ├── sampling.py            # Sample G outputs
│   │   ├── reward.py              # Compute R_i
│   │   ├── advantage.py           # (R - μ) / σ^eff
│   │   ├── clipping.py            # clip(ρ, 1-ε_l, 1+ε_h)
│   │   ├── normalization.py       # 1/|o_i| or 1/Σ|o_i|
│   │   ├── aggregation.py         # Σ over tokens/responses
│   │   ├── diversity.py           # R̃ = R · (1 - SMI)
│   │   ├── lambda_weight.py       # f_λ(o_i)
│   │   ├── difficulty_weight.py   # w_μ per group
│   │   ├── filtering.py           # I[0 < μ_q < 1]
│   │   └── kl_penalty.py          # -β·D_KL
│   │
│   ├── builder/                    # Objective construction
│   │   ├── __init__.py
│   │   ├── objective_builder.py   # Main factory
│   │   ├── graph.py               # Computation graph representation
│   │   └── registry.py            # Operation registry
│   │
│   ├── objective/                  # Final objective modules
│   │   ├── __init__.py
│   │   ├── unified_objective.py   # UnifiedObjective(nn.Module)
│   │   └── components.py          # ComposableComponent base class
│   │
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py             # Training loop (uses objective)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tensor_ops.py
│       └── metrics.py
│
├── configs/                        # YAML configs
│   ├── grpo.yaml                  # Pure GRPO
│   ├── dapo.yaml                  # Pure DAPO
│   ├── hybrid_dra_lambda.yaml     # DRA + λ-GRPO
│   └── custom.yaml                # User's custom config
│
├── examples/
│   ├── 01_train_grpo.py
│   ├── 02_train_hybrid.py
│   ├── 03_custom_objective.py
│   └── 04_ablation_study.py
│
└── tests/
    ├── test_operations/
    ├── test_builder/
    └── test_recovery.py           # Verify config → correct objective as project structure.
