

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# ─── Interface 2: TrainingEngine ────────────────────────────
class TrainingEngine(ABC):
    """With grad. Wraps nn.Module + distributed + optimizer."""
    @abstractmethod
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """Forward pass WITH grad → log-probs for loss computation."""
        pass

    @abstractmethod
    def backward(self, loss) -> None:
        """Backward + gradient sync across ranks."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Optimizer step + LR scheduler."""
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Access underlying nn.Module (for weight sync, checkpointing)."""
        pass
