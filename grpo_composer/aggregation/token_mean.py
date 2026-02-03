"""
Base GRPO: Per-Sequence Token Mean Aggregation

Paper: DeepSeekMath (Base GRPO)

Mathematical Form:
    L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

This is the standard GRPO aggregation:
- First average tokens within each sequence: (1/|o_i|) * Σ_t
- Then average across group: (1/G) * Σ_i

Effect:
    Each sequence contributes equally regardless of length.
    Longer sequences have diluted per-token gradients.
"""
