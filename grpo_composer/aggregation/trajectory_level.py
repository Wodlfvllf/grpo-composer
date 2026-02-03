"""
TIC-GRPO: Trajectory-Level Aggregation (Not Token-Level)

Paper: TIC-GRPO

Components Changed (from base GRPO):
- Standard GRPO: Per-token ratios P_θ(s_t|s_{t-1}) / P_θ_old(s_t|s_{t-1})
- TIC-GRPO: Single trajectory ratio P_θ(s_T|s_0) / P_θ_old(s_T|s_0)

Mathematical Form:
    Standard GRPO (token-level):
        ρ_{i,t} = π_θ(o_{i,t}|q, o_{i,<t}) / π_θ_old(o_{i,t}|q, o_{i,<t})

    TIC-GRPO (trajectory-level):
        ρ_i = P_θ(s_T|s_0) / P_θ_old(s_T|s_0)
            = Π_t [π_θ(o_{i,t}|...) / π_θ_old(o_{i,t}|...)]

Benefit:
    Better convergence rate O(log|V| / √N) without MN or σ²_{sT} terms.
    Uses upper-only clipping at trajectory level.
"""
