
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
    def __init__(self, rewards: torch.tensor, embedding: torch.tensor, **kwargs):
        super().__init__(rewards, **kwargs)
        self.rewards = rewards
        self.embedding = embedding
        self.kwargs = kwargs

        assert rewards.ndim == 2, "Rewards should be a 2D tensor"
        assert embedding.ndim == 3, "Embedding should be a 3D tensor"
        assert rewards.shape[0] == embedding.shape[0], "Batch size mismatch"
        assert rewards.shape[1] == embedding.shape[1], "Group size mismatch"

    def _compute_similarity(self):
        """
        Compute pairwise cosine similarity matrix for one prompt's group.
        
        Args:
            embeddings: torch.Tensor, shape (num_completions, hidden_size)
        Returns:
            sim_matrix: torch.Tensor, shape (num_completions, num_completions)
        """
        embeddings_norm = torch.linalg.norm(self.embedding, dim=-1, keepdim=True)
        #shape output - (num_completions(group_size), 1)
        embeddings = embeddings - embeddings_norm
        return embeddings @ embeddings.T

    def _calculate_smi(self):
        """
        Calculate Submodular Mutual Information (SMI) for all completions.
        
        SMI(o_i) = sum_{j != i} similarity(o_i, o_j)
        
        Returns:
            smi_scores: torch.Tensor, shape (batch_size, num_completions)
        """

        # shape of the self.rewards is (Batch_size, num_completions(group_size))
        # there would be Batch_size number of smi calculations for each prompt.
        # Also SMI would be calculated on Embeddings and then rewards would be calculated on top of it.
        # Embedding shape is (Batch_size, num_completions(group_size), hidden_size)
        batch_size, num_completions, hidden_size = self.embedding.shape
        # Initialize tensor to store SMI scores
        smi_scores = torch.zeros(batch_size, num_completions, device=self.rewards.device)

        # Process each batch independently
        for batch_idx in range(batch_size):
            # Get embeddings for this prompt's group
            prompt_embeddings = self.embedding[batch_idx]  # (num_completions, hidden_size)

            # Compute similarity matrix
            sim_matrix = self._compute_similarity(prompt_embeddings)

            # Calculate SMI for this prompt
            smi_scores[batch_idx] = torch.sum(sim_matrix, dim = -1) - torch.diagonal(sim_matrix)
            
        return smi_scores

    def calculate(self):
        self._calculate_smi()
        return self.rewards