"""
TR-GRPO: Probability-Weighted Token Aggregation for Sharpness Control

Paper: TR-GRPO

Components Changed (from base GRPO):
- Each token weighted by its probability under current policy
- Down-weights high-probability tokens, reduces gradient norm/sharpness

Mathematical Form:
    Token weight:
        w_{i,t} = clip(α * (σ(π_θ(o_{i,t}) / τ) - μ), L, U)

    Weighted gradient:
        g_{i,t} = γ_{i,t} * w_{i,t} * ∇_θ log π_θ(o_{i,t})

    Where:
        σ = sigmoid function
        τ = temperature
        μ = centering constant
        L, U = clipping bounds

Effect:
    Down-weights already-confident tokens.
    Reduces sharpness and gradient norm.
    More stable training.
"""
