"""
λ-GRPO: Learnable Response-Level Weights Based on Length

Paper: λ-GRPO

Components Changed (from base GRPO):
- Replaces uniform 1/G with learnable weights f_λ(o_i)
- Power-law formulation based on normalized length

Mathematical Form:
    Standard GRPO:
        L = (1/G) * Σ_i L_i   (uniform weighting)

    λ-GRPO:
        L = Σ_i f_λ(o_i) * L_i

    Weight computation:
        μ_ℓ = (1/G) * Σ_j |o_j|           # Mean length
        σ_ℓ = std({|o_j|})                 # Length std
        z_i = (|o_i| - μ_ℓ) / (σ_ℓ + ε)   # Standardize
        h_i = 1 + r * z_i                  # Shift
        f_λ(o_i) = G * softmax(h_i^λ)      # Power-law + normalize

    λ values:
        λ = 0  → Uniform (DAPO-like)
        λ < 0  → Penalize length (GRPO-like)
        λ > 0  → Reward length
        λ learnable → Adaptive
"""
