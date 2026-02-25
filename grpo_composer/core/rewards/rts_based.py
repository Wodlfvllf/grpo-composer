"""
SPO: Spectral Policy Optimization — RLAIF Reward with Reasoning Trajectory Score

Paper: "SPO: Coloring your Incorrect Reasoning in GRPO"

Uses a Reasoning Trajectory Score (RTS) to give partial credit for
correct reasoning segments in WRONG answers. RTS is computed externally
(by prompting a stronger model to identify the first mistake).

Formula (Paper Eq. 2):
    r_AIF(y) = 1,                                    if y is correct
             = 1 / (1 + exp(β × (RTS(y) - γ))),     otherwise

Where:
    - RTS(y) ∈ [0, 1]: ratio of correct reasoning length to total length
    - β > 0: scale intensity (sharper sigmoid)
    - γ > 0: scale threshold (shift inflection point)

This ensures incorrect answers with more correct reasoning get higher
(but still < 1) rewards, providing informative gradient signal even for
all-negative groups.

Input:
    correctness: (B, G) binary correctness {0, 1}
    rts_scores: (B, G) pre-computed RTS values ∈ [0, 1]
Output:
    (B, G) RLAIF rewards
"""

import torch


class RTSRewardCalculator:
    """
    SPO: RLAIF reward with Reasoning Trajectory Score.

    Args:
        beta: Scale intensity — larger = sharper sigmoid (default: 5.0)
        gamma: Scale threshold — sigmoid inflection point (default: 0.5)
    """

    def __init__(self, beta: float = 5.0, gamma: float = 0.5):
        self.beta = beta
        self.gamma = gamma

    def compute_rewards(
        self,
        correctness: torch.Tensor,
        rts_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RLAIF rewards from correctness and RTS scores.

        Args:
            correctness: (B, G) binary {0, 1}
            rts_scores: (B, G) RTS values ∈ [0, 1] (pre-computed externally)

        Returns:
            rewards: (B, G) RLAIF rewards

        For correct answers: r = 1
        For wrong answers: r = 1 / (1 + exp(β × (RTS - γ)))
            High RTS (good reasoning, wrong answer): reward → sigmoid value
            Low RTS (bad reasoning, wrong answer): reward → lower value
        """
        # Sigmoid reward for wrong answers
        sigmoid_reward = 1.0 / (1.0 + torch.exp(self.beta * (rts_scores - self.gamma)))

        # Correct → 1, Wrong → sigmoid(RTS)
        rewards = torch.where(
            correctness > 0.5,
            torch.ones_like(correctness),
            sigmoid_reward,
        )

        return rewards
