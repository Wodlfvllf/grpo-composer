"""
XRPO: Novelty-Guided Advantage Sharpening for Exploitation

Paper: XRPO

Components Changed (from base GRPO):
- Adds novelty bonus to advantage for novel yet correct sequences
- Expands policy boundary by boosting rare correct solutions

Mathematical Form:
    Log-likelihood score:
        s(y) = (1/|y|) * Σ_t log π_θ(y_t | x, y_{<t})

    Novelty (relative to group average):
        η_i = exp(s(y_i) - s̄)

    Advantage sharpening:
        A⁺_i = A_i + min(max(λ_novelty * (1 - η_i), 0), κ_clip * A_i)

Effect:
    Boosts novel yet correct sequences, expands policy boundary
"""
