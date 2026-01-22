"""
Diversity Adjusted Reward Calculator
This class implements Diversity-aware Reward Adjustment from DRA-GRPO paper.

R(q, o_i) = R(q, o_i) / (1 + SMI({o_i}, C \\ {o_i}))

Where SMI({o_i}, C \\ {o_i}) denotes the Submodular Mutual Information (SMI) between
query completion o_i and the remaining completions C \\ {o_i}. Submodular functions,
with their diminishing returns property, naturally model diversity and redundancy.
SMI quantifies the shared information between sets under a submodular function
(Iyer et al., 2021a,b).

We instantiate SMI using the Graph-Cut function over a similarity kernel s(·,·):
SMI({o_i}, C \\ {o_i}) = Σ_{j ∈ C \\ {o_i}} s(o_i, j)
""" 

import torch
import torch.nn.functional as F
from .base import RewardCalculator


class DiversityAdjustedRewardCalculator(RewardCalculator):
    """
    DRA-GRPO: Diversity-Aware Reward Adjustment using Submodular Mutual Information.
    
    Formula: adjusted_reward = reward / (1 + SMI)
    where SMI(o_i) = sum_{j != i} similarity(o_i, o_j)
    """
    
    def __init__(self, rewards: torch.Tensor, embedding: torch.Tensor, epsilon: float = 1e-6, **kwargs):
        """
        Args:
            rewards: torch.Tensor, shape (batch_size, num_completions)
            embedding: torch.Tensor, shape (batch_size, num_completions, hidden_size)
            epsilon: Small constant for numerical stability (default: 1e-6)
        """
        super().__init__(rewards, **kwargs)
        self.rewards = rewards
        self.embedding = embedding
        self.epsilon = epsilon
        self.kwargs = kwargs
        
        # Validate shapes
        assert rewards.ndim == 2, f"Rewards must be 2D, got shape {rewards.shape}"
        assert embedding.ndim == 3, f"Embedding must be 3D, got shape {embedding.shape}"
        assert rewards.shape[0] == embedding.shape[0], \
            f"Batch size mismatch: rewards {rewards.shape[0]} vs embeddings {embedding.shape[0]}"
        assert rewards.shape[1] == embedding.shape[1], \
            f"Group size mismatch: rewards {rewards.shape[1]} vs embeddings {embedding.shape[1]}"

    def _compute_similarity(self, embeddings):
        """
        Compute pairwise cosine similarity matrix for one prompt's group.
        
        Args:
            embeddings: torch.Tensor, shape (num_completions, hidden_size)
        Returns:
            sim_matrix: torch.Tensor, shape (num_completions, num_completions)
        """
        # Normalize embeddings to unit vectors (L2 normalization)
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        # Shape: (num_completions, hidden_size)
        
        # Compute cosine similarity via dot product of normalized vectors
        sim_matrix = embeddings_norm @ embeddings_norm.T
        # Shape: (num_completions, num_completions)
        
        return sim_matrix

    def _calculate_smi(self):
        """
        Calculate Submodular Mutual Information (SMI) for all completions.
        
        SMI(o_i) = sum_{j != i} similarity(o_i, o_j)
        
        Returns:
            smi_scores: torch.Tensor, shape (batch_size, num_completions)
        """
        batch_size, num_completions, hidden_size = self.embedding.shape
        
        # Initialize tensor to store SMI scores
        smi_scores = torch.zeros(batch_size, num_completions, device=self.rewards.device)
        
        # Process each batch independently
        for batch_idx in range(batch_size):
            # Get embeddings for this prompt's group
            prompt_embeddings = self.embedding[batch_idx]  # (num_completions, hidden_size)
            
            # Compute similarity matrix
            sim_matrix = self._compute_similarity(prompt_embeddings)
            # Shape: (num_completions, num_completions)
            
            # Calculate SMI: sum of row - diagonal (exclude self-similarity)
            smi_scores[batch_idx] = torch.sum(sim_matrix, dim=-1) - torch.diagonal(sim_matrix)
            # Shape: (num_completions,)
        
        return smi_scores

    def calculate(self):
        """
        Apply DRA-GRPO diversity adjustment.
        
        Returns:
            adjusted_rewards: torch.Tensor, shape (batch_size, num_completions)
        """
        # Compute SMI scores for all completions
        smi_scores = self._calculate_smi()
        # Shape: (batch_size, num_completions)
        
        # Apply DRA-GRPO formula: adjusted = reward / (1 + SMI)
        adjusted_rewards = self.rewards / (1.0 + smi_scores + self.epsilon)
        # Shape: (batch_size, num_completions)
        
        return adjusted_rewards
