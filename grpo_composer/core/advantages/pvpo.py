import torch
from typing import Dict, List, Optional
from grpo_composer.core.advantages.base import AdvantageFunction

class StaticValueAdvantageFunction(AdvantageFunction):
    """
    Computes PVPO-style advantages.
    Â_i = r_i - μ(r_ref)
    
    The baseline is the mean of the static reference model's rewards for the prompt,
    bypassing the need for a dynamically trained Value network.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def compute_advantages(
        self, 
        rewards: torch.Tensor,
        reference_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rewards: (1, G) tensor of actor rewards for a single prompt group
            reference_rewards: (1, G) tensor of reference rewards for the same prompt group
        """
        advantages = torch.zeros_like(rewards, dtype=torch.float32)

        mean_ref_reward = reference_rewards.mean(dim=-1, keepdim=True)
        
        # PVPO specifically does NOT normalize by std dev! It subtracts the static mean.
        advantages = rewards - mean_ref_reward
            
        return advantages
