
"""
Diversity Adjusted Reward Calculator
This class implements Diversity-aware Reward Adjustment from DRA-GRPO paper.

R(q, o_i) = R(q, o_i) / (1 + SMI({o_i}, C \ {o_i}))

Where SMI({o_i}, C \ {o_i}) denotes the Submodular Mutual Information (SMI) between
query completion o_i and the remaining completions C \ {o_i}. Submodular functions,
with their diminishing returns property, naturally model diversity and redundancy.
SMI quantifies the shared information between sets under a submodular function
(Iyer et al., 2021a,b).

We instantiate SMI using the Graph-Cut function over a similarity kernel s(·,·):
SMI({o_i}, C \ {o_i}) = Σ_{j ∈ C \ {o_i}} s(o_i, j)
""" 

import torch
from .base import RewardCalculator


class DiversityAdjustedRewardCalculator(RewardCalculator):
    def __init__(self, rewards: list, **kwargs):
        super().__init__(rewards, **kwargs)
        self.rewards = rewards
        self.kwargs = kwargs

    def _calculate_smi(self):
        """
        Calculate Submodular Mutual Information (SMI) between query completion o_i and the remaining completions C \ {o_i}.
        """
        # shape of the self.rewards is (Batch_size, num_completions)
        # there would be Batch_size number of smi calculations for each prompt.
        batch_size, max_seq_len = self.rewards.shape
        
        for indx in range(batch_size):
            rewards = self.rewards[indx]
            smi_matrix = (rewards.T @ rewards)
            smi_score = torch.sum(smi_matrix) - smi_matrix[indx][indx]
            rewards[indx] = rewards[indx] / (1 + smi_score)
            self.rewards[indx] = rewards

    def calculate(self):
        self._calculate_smi()
        return self.rewards