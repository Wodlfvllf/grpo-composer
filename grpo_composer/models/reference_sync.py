"""
Reference Model Sync Strategies

Controls how the reference model stays in sync with the policy model.
In GRPO, the reference model provides the baseline log-probs for KL divergence.

Strategies:
    - FrozenSync:   Never updates (default GRPO — ref = initial policy)
    - EMASync:      Exponential moving average of policy weights
    - PeriodicSync: Copy policy weights every N steps
"""

import torch
import torch.nn as nn
from copy import deepcopy


class FrozenSync:
    """Reference model is a frozen copy. Never updated."""

    def __init__(self, policy_model: nn.Module):
        self.ref_model = deepcopy(policy_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def step(self, policy_model: nn.Module, global_step: int):
        pass  # no-op

    def get_model(self) -> nn.Module:
        return self.ref_model


class EMASync:
    """Reference model updated via exponential moving average.
    
    ref_params = decay * ref_params + (1 - decay) * policy_params
    """

    def __init__(self, policy_model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ref_model = deepcopy(policy_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def step(self, policy_model: nn.Module, global_step: int):
        for ref_p, pol_p in zip(self.ref_model.parameters(), policy_model.parameters()):
            ref_p.data.mul_(self.decay).add_(pol_p.data, alpha=1.0 - self.decay)

    def get_model(self) -> nn.Module:
        return self.ref_model


class PeriodicSync:
    """Reference model updated by copying policy weights every N steps."""

    def __init__(self, policy_model: nn.Module, sync_every: int = 100):
        self.sync_every = sync_every
        self.ref_model = deepcopy(policy_model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def step(self, policy_model: nn.Module, global_step: int):
        if global_step > 0 and global_step % self.sync_every == 0:
            self.ref_model.load_state_dict(policy_model.state_dict())

    def get_model(self) -> nn.Module:
        return self.ref_model
