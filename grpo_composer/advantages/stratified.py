"""
Stratified-GRPO: Per-Stratum Advantage Normalization (SAN)

Paper: Stratified-GRPO

Components Changed (from base GRPO):
- Partitions trajectories into strata by structure (e.g., search count)
- Normalizes advantages WITHIN each stratum, not globally

Mathematical Form:
    Problem: Global normalization has cross-stratum bias when trajectories 
             differ structurally

    Partition: Group trajectories into strata by structure

    Per-Stratum Normalization (SAN):
        A^{SAN}(τ_i) = (R(τ_i) - μ̃_k) / (σ̃_k + ε)
        
        Where μ̃_k, σ̃_k are computed within stratum k only

    Blended (for finite-sample stability):
        A^{blend} = α * A^{SAN} + (1 - α) * A^{GN}

Benefit:
    Eliminates between-stratum variance, zero conditional bias per stratum
"""
