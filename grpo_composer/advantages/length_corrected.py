"""
TIC-GRPO: Trajectory-level Importance-Corrected, Length-Corrected Group Normalization

Paper: TIC-GRPO

Components Changed (from base GRPO):
- Reward normalized by sequence length BEFORE advantage calculation
- Better convergence rate without σ²_{sT} terms

Mathematical Form:
    Standard GRPO:
        A_G(s_T) = (r(s_T) - μ_G) / (σ_G + δ)

    TIC-GRPO (length-corrected):
        A'_G(s_T) = (r(s_T) / |s_T| - μ'_G) / (σ'_G + δ)

    Where μ'_G and σ'_G are computed on length-normalized rewards.

Benefit:
    Removes dependence on sequence length variance σ²_{sT,N}
"""
