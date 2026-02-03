"""
DAPO: Global Token Normalization

Paper: DAPO

Components Changed (from base GRPO):
- Instead of per-sequence 1/|o_i|, uses global token count
- Normalizes by TOTAL tokens across all sequences in group

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i (1/|o_i|) * Σ_t loss_{i,t}

    DAPO:
        L = (1 / Σ_i |o_i|) * Σ_i Σ_t loss_{i,t}

Rationale:
    Each TOKEN contributes equally, regardless of which sequence it's in.
    Avoids bias toward shorter sequences.
"""
