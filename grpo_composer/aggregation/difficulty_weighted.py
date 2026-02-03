"""
DARO: Learnable Per-Difficulty Weights for Curriculum Balancing

Paper: DARO

Components Changed (from base GRPO):
- Groups prompts by difficulty μ_q = (#correct) / G
- Learnable weight w_μ per difficulty bin
- Optimal weights inversely proportional to loss

Mathematical Form:
    Standard GRPO:
        L = uniform average across all prompts

    DARO:
        L = Σ_{μ∈M} [w_μ * L_μ(θ) - ln(w_μ)]

    Difficulty bins:
        M = {1/G, 2/G, ..., (G-1)/G}
        Excludes μ=0 and μ=1 (zero gradient)

    Optimal weights:
        w*_μ ∝ L_μ^{-1}   (inverse loss weighting)

    Per-group normalization:
        Ω_μ = 1/L_μ where L_μ = total tokens in difficulty group

Effect:
    Balances learning across difficulty levels.
    Prevents easy/hard prompts from dominating gradients.
"""
