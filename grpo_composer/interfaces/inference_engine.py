

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RolloutRequest:
    input_ids : torch.Tensor # Shape (B * G, T)
    attention_mask : torch.Tensor # Shape (B * G, T)

@dataclass
class RolloutResult:
    completions : torch.Tensor   # (B*G, L) — generated token IDs, padded
    log_probs : torch.Tensor     # (B*G, L) — per-token log-probs

# ─── Interface 1: InferenceEngine ───────────────────────────
class InferenceEngine(ABC):
    """No grad. Used for generation and log-prob scoring."""

    @abstractmethod
    def generate(self, request : RolloutRequest) -> RolloutResult:
        """Autoregressive loop → completions + log_probs."""
        pass

    @abstractmethod
    def get_log_probs(self, input_ids, attention_mask) -> torch.Tensor:
        """Single forward pass → log-probs of existing tokens."""
        pass