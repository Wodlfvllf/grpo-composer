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

class LengthCorrectedAdvantageFunction(AdvantageFunction):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def compute_advantages(self, rewards, lengths):
        """
        TIC-GRPO advantage computation (length-corrected).

        Args:
            rewards: list or np.array of shape [G]
                    binary rewards per trajectory
            lengths: list or np.array of shape [G]
                    number of tokens per trajectory

        Returns:
            advantages: np.array of shape [G]
                        one advantage per trajectory
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.float32)

        # 1. Convert reward -> reward per token
        reward_per_token = rewards / lengths

        # 2. Group normalization
        mean = reward_per_token.mean()
        std = reward_per_token.std() + self.epsilon

        advantages = (reward_per_token - mean) / std
        return advantages
