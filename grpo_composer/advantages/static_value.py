"""
PVPO: Static Value Estimate from Reference Policy

Paper: PVPO

Components Changed (from base GRPO):
- Baseline: Uses STATIC reference mean instead of dynamic group mean
- Decouples Q (from on-policy) and V (from reference)

Mathematical Form:
    Standard GRPO (dynamic baseline):
        Â = r_i - mean(r)              # Dynamic group mean

    PVPO (static baseline):
        Â^{PVPO} = r_i - mean(r_ref)   # Static reference mean

    Where:
        Q (dynamic): From on-policy π_θ rollout reward
        V (static): Pre-estimated from reference π_ref: V̂_sta = (1/M) * Σ_j r^{ref}_j

Benefit:
    Stable learning signal even under severe reward sparsity
"""
