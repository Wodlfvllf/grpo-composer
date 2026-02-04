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

class StratifiedAdvantageFunction(AdvantageFunction):
    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def compute_advantages(self, rewards: torch.Tensor, strata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (B, G) rewards
            strata: (B, G) strata
        Returns:
            advantages: (B, G) stratified advantages
        """
        B, G = rewards.shape
        advantage = torch.zeros_like(rewards)

        for idx in range(B):
            reward = rewards[idx]
            stratum = strata[idx]
            
            for strata in set(stratum):
                mask = (stratum == strata)
                mean = reward[mask].mean()
                std = reward[mask].std() + self.epsilon
                advantage[idx][mask] = (reward[mask] - mean) / std
        
        base_advantage = (rewards - rewards.mean(dim=-1, keepdim=True)) / (rewards.std(dim=-1, keepdim=True) + self.epsilon)
        return self.alpha * advantage + (1 - self.alpha) * base_advantage