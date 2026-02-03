"""
KRPO: Kalman Filter-Based Advantage Estimation

Paper: KRPO

Components Changed (from base GRPO):
- Mean: Replaced with Kalman Filter state estimate x̂_{t|t}
- Std Dev: Replaced with √P_{t|t} (uncertainty variance)
- Reward: No change

Mathematical Form:
    A_i = (r_i - x̂_{t|t}) / (√P_{t|t} + ε)

Where:
    x̂_{t|t} = Kalman Filter state estimate (dynamic baseline)
    P_{t|t} = Posterior covariance (uncertainty)
"""

import torch
from .base import AdvantageFunction
class KalmanAdvantageFunction(AdvantageFunction):
    def __init__(
        self, 
        process_noise: float = 1e-4,    # Q
        measurement_noise: float = 1.0,  # R
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.Q = process_noise
        self.R = measurement_noise
        self.epsilon = epsilon
        # State initialized on first call
        self.x_hat = None  # State estimate
        self.P = None      # Error covariance
    def _kalman_update(self, measurement: float):
        """Single Kalman filter update step."""
        if self.x_hat is None:
            # Initialize on first observation
            self.x_hat = measurement
            self.P = 1.0
            return
        
        # Predict step (constant dynamics: x_{t+1} = x_t)
        x_pred = self.x_hat
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred / (P_pred + self.R)           # Kalman gain
        self.x_hat = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rewards: (B, G) rewards
        Returns:
            advantages: (B, G)
        """
        # Update Kalman filter with batch mean
        batch_mean = rewards.mean().item()
        self._kalman_update(batch_mean)
        
        # Advantage = (r - x̂) / √P
        advantages = (rewards - self.x_hat) / (torch.sqrt(torch.tensor(self.P)) + self.epsilon)
        
        return advantages