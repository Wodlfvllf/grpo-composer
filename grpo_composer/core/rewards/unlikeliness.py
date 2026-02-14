

"""
Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening

Mathematical Form
-----------------

For a group of samples {y_1, ..., y_G}:

1. Rank samples by probability under the current policy
2. Let rank(y_i) = 0 for most likely, larger for rarer
3. Modify the reward as:

    r_i = R(x, y_i) * (1 + β * rank(y_i) / G)

Important Details:
- Incorrect samples still get zero reward
- Only correct samples are reweighted
- Rarer correct samples get larger effective advantage
- The GRPO objective itself is unchanged

This is called Unlikeliness Reward.
"""

import torch

class UnlikelinessRewardCalculator:
    def __init__(self, beta=0.5):
        self.beta = beta

    def compute_rewards(self, rewards, log_probs):
        """
        Args:
            rewards: torch.Tensor, shape (B, G), 0 for incorrect
            log_probs: torch.Tensor, shape (B, G)
        Returns:
            torch.Tensor, shape (B, G)
        """
        B, G = rewards.shape

        # Rank by probability (higher prob = lower rank)
        ranks = torch.argsort(
            torch.argsort(-log_probs, dim=1),
            dim=1
        )  # 0 = most likely

        # Normalize rank
        rank_frac = ranks.float() / (G - 1)

        # Increase reward for rarer correct samples
        weights = 1.0 + self.beta * rank_frac

        # Only scale correct samples
        return rewards * weights

        