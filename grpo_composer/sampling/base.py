"""
Base Abstract Class for Prompt Sampling in GRPO

Sampling controls which prompts are included in training:
- Standard: All prompts equally
- Dynamic (DAPO): Filter uninformative prompts (μ=0 or μ=1)
- Difficulty-Grouped (DARO): Group by difficulty bins
- GT Injection (PVPO): Inject ground truth for zero-accuracy prompts
- Hierarchical (XRPO): Exploration-exploitation with uncertainty
"""
