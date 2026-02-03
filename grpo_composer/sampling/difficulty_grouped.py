"""
DARO: Difficulty-Grouped Sampling

Paper: DARO

Components Changed (from base GRPO):
- Groups prompts by difficulty level μ_q
- Excludes μ=0 and μ=1 (zero gradient)
- Per-group normalization

Mathematical Form:
    Difficulty bins:
        M = {1/G, 2/G, ..., (G-1)/G}

    For prompt q:
        μ_q = (# correct responses) / G
        Assign q to bin μ_q

    Exclude:
        μ = 0 (all wrong) → zero gradient
        μ = 1 (all correct) → zero gradient

    Per-group normalization:
        Ω_μ = 1/L_μ
        Where L_μ = total tokens in difficulty group μ

Effect:
    Balanced sampling across difficulty levels.
    Combined with learnable weights in aggregation.
"""
