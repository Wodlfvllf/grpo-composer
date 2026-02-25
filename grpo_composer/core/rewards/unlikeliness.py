"""
Unlikeliness Reward Calculator (Rewarding the Unlikely)

Paper: "Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening"

Modifies rewards with a rank-based penalty that penalizes high-probability
correct solutions and boosts low-probability ones. Counters GRPO's tendency
toward distribution sharpening (only boosting already-likely solutions).

Formula (Paper Eq. 13):
    r_i = R(x, y_i) × (1 - β_rank × (G - rank(y_i)) / G)

Where:
    - R(x, y_i) ∈ {0, 1}: binary correctness
    - rank(y_i) ∈ {1, ..., G}: rank by π_old probability (1 = most likely)
    - β_rank > 0: controls strength of rank perturbation

Effect:
    - rank 1 (most likely):  penalty ≈ β_rank     (highest penalty)
    - rank G (least likely): penalty = 0           (no penalty)
    - Incorrect solutions:   r_i = 0 regardless

Input:
    rewards: (B, G) binary correctness
    log_probs: (B, G) sequence-level log-probs from π_old
Output:
    (B, G) rank-adjusted rewards
"""

import torch


class UnlikelinessRewardCalculator:
    """
    Rewarding the Unlikely: rank-based reward modification.

    Args:
        beta_rank: Strength of rank penalty (default: 0.5)
    """

    def __init__(self, beta_rank: float = 0.5):
        self.beta_rank = beta_rank

    def compute_rewards(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rewards: (B, G) binary correctness rewards
            log_probs: (B, G) sequence-level log-probs from π_old
        Returns:
            (B, G) rank-adjusted rewards
        """
        B, G = rewards.shape

        # Rank by probability: higher log_prob = rank 1 (most likely)
        ranks_0indexed = torch.argsort(
            torch.argsort(-log_probs, dim=1),
            dim=1,
        )
        ranks = (ranks_0indexed + 1).float()  # 1-indexed per paper

        # Paper Eq. 13: r_i = R(x,y) × (1 - β_rank × (G - rank) / G)
        penalty = (G - ranks) / G
        return rewards * (1.0 - self.beta_rank * penalty)