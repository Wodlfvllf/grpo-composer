"""
GRPO-LEAD: Length-Dependent Reward + Difficulty-Aware Advantage Reweighting

Paper: GRPO-LEAD

Components Changed (from base GRPO):
- Reward: Length-penalized accuracy reward
- Advantage: Difficulty-aware reweighting

Mathematical Form:
    Length-penalized reward:
        z = (|o| - μ) / (σ + ε)                   # Standardized length
        R_acc(o|q) = exp(-α*z)  if correct
                   = -1         if incorrect

    Difficulty proxy:
        ρ_q = (# correct) / (# total)

    Logistic weight:
        w(ρ_q) = A + (B - A) / (1 + exp[k(ρ_q - ρ_0)])

    Difficulty-aware advantage (ASYMMETRIC):
        A'_i = Ã_i * w(ρ_q)      if Ã_i > 0    ← positive: weight by difficulty
             = Ã_i * w(1 - ρ_q)  if Ã_i ≤ 0    ← negative: weight by EASINESS

    NOTE: w(1 - ρ_q) means re-evaluate the logistic at (1-ρ), NOT 1 - w(ρ).
    For hard problems (low ρ): w(ρ) is high → boost correct.
    For easy problems (high ρ): w(1-ρ) is high → penalize incorrect harder.

Paper Hyperparameters:
    A = 0.4, B = 1.5, ρ_0 = 0.75, k = 10
    (no KL penalty — found to suppress exploration)
"""

import torch
from .base import AdvantageFunction


class DifficultyAwareAdvantageFunction(AdvantageFunction):
    def __init__(self, A: float = 0.4, B: float = 1.5, k: float = 10.0,
                 rho_0: float = 0.75, epsilon: float = 1e-8):
        super().__init__()
        self.A = A
        self.B = B
        self.k = k
        self.rho_0 = rho_0
        self.epsilon = epsilon

    def _logistic_weight(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Logistic reweighting: w(ρ) = A + (B - A) / (1 + exp[k(ρ - ρ_0)])

        Args:
            rho: (B,) correctness ratios per question
        Returns:
            weights: (B,) logistic weights
        """
        return self.A + (self.B - self.A) / (1 + torch.exp(self.k * (rho - self.rho_0)))

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute difficulty-aware advantages.

        Args:
            rewards: (B, G) rewards per response

        Returns:
            advantages: (B, G) difficulty-reweighted advantages

        Shape Flow:
            rewards: (B, G)
            difficulty_proxy: (B,) — ρ_q = mean(rewards > 0)  [empirical correctness]
            logistic_weight_pos: (B, 1) — w(ρ_q) for positive advantages
            logistic_weight_neg: (B, 1) — w(1 - ρ_q) for negative advantages
            base_advantage: (B, G) — standard group normalization
            difficulty_aware_advantage: (B, G) — asymmetric reweighting
        """
        # Difficulty proxy: ρ_q = fraction of correct responses
        # Use (reward > 0) as proxy for correctness (paper uses binary rewards)
        difficulty_proxy = (rewards > 0).float().mean(dim=-1)  # (B,)

        # Logistic weights — ASYMMETRIC per paper Eq. 9
        # w(ρ) for positive advantages (boost rare correct on hard problems)
        w_pos = self._logistic_weight(difficulty_proxy).unsqueeze(-1)     # (B, 1)
        # w(1 - ρ) for negative advantages (penalize incorrect on easy problems)
        w_neg = self._logistic_weight(1.0 - difficulty_proxy).unsqueeze(-1)  # (B, 1)

        # Base advantage: standard group normalization
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True) + self.epsilon
        base_advantage = (rewards - mean) / std  # (B, G)

        # Asymmetric difficulty-aware advantage
        # Positive advantages: A'_i = Ã_i * w(ρ_q)
        # Negative advantages: A'_i = Ã_i * w(1 - ρ_q)
        difficulty_aware_advantage = torch.where(
            base_advantage > 0,
            base_advantage * w_pos,    # w(ρ) — high for hard problems
            base_advantage * w_neg     # w(1-ρ) — high for easy problems
        )

        return difficulty_aware_advantage