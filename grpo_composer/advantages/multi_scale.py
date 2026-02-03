"""
MS-GRPO: Multi-Scale Advantage via Hierarchical Subgroup Comparisons

Paper: MS-GRPO

Components Changed (from base GRPO):
- Advantage computed at MULTIPLE scales τ ∈ {τ_min, ..., G}
- Final advantage is weighted average across scales

Mathematical Form:
    Single-scale advantage for subgroup S:
        Â_{i,t}(S) = (r_{i,t} - μ_S) / σ_S

    Scale-specific average:
        Ā^(τ)_{i,t} = (1 / C(G-1, τ-1)) * Σ_{|S|=τ, o_i∈S} Â_{i,t}(S)

    Multi-scale advantage:
        Â^{MS-GRPO}_{i,t} = Σ_{τ=τ_min}^G w_τ * Ā^(τ)_{i,t}

    Weights: w_τ ≥ 0, Σ w_τ = 1 (default: uniform)
"""
