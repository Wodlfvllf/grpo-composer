"""
Dr.GRPO: Token Sum Aggregation (No Length Normalization)

Paper: Dr.GRPO

Components Changed (from base GRPO):
- Removes 1/|o_i| per-sequence length normalization
- Sums tokens instead of averaging

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

    Dr.GRPO:
        L = (1/G) * Σ_i Σ_t loss_{i,t}

Rationale:
    Avoids gradient dilution for longer sequences.
    Longer correct responses get proportionally more gradient signal.
"""
