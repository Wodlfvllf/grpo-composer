"""
Diversity Adjusted Reward Calculator (DRA-GRPO)

Implements Diversity-aware Reward Adjustment using Submodular Mutual Information.

Formula:
    R'(q, o_i) = R(q, o_i) / (1 + SMI({o_i}, C \\ {o_i}))

Where SMI is computed as the sum of cosine similarities to other completions:
    SMI({o_i}, C \\ {o_i}) = Σ_{j ≠ i} cos_sim(embedding_i, embedding_j)

Input:
    rewards: torch.Tensor of shape (batch_size, num_completions)
    embedding: torch.Tensor of shape (batch_size, num_completions, hidden_size)
    
Output:
    torch.Tensor of shape (batch_size, num_completions)
"""

import torch
import torch.nn.functional as F
from .base import RewardCalculator


class DiversityAdjustedRewardCalculator(RewardCalculator):
    """
    DRA-GRPO: Diversity-Aware Reward Adjustment using Submodular Mutual Information.
    
    Penalizes redundant responses by dividing reward by (1 + SMI), where SMI
    measures similarity to other responses in the group.
    
    Args:
        rewards: torch.Tensor, shape (batch_size, num_completions)
        embedding: torch.Tensor, shape (batch_size, num_completions, hidden_size)
        epsilon: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(
        self, 
        rewards: torch.Tensor, 
        embedding: torch.Tensor, 
        epsilon: float = 1e-6, 
        **kwargs
    ) -> None:
        super().__init__(rewards, **kwargs)
        self.embedding = embedding
        self.epsilon = epsilon
        
        # Validate shapes
        if rewards.ndim != 2:
            raise ValueError(f"Rewards must be 2D, got shape {rewards.shape}")
        if embedding.ndim != 3:
            raise ValueError(f"Embedding must be 3D, got shape {embedding.shape}")
        if rewards.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"Batch size mismatch: rewards {rewards.shape[0]} vs embeddings {embedding.shape[0]}"
            )
        if rewards.shape[1] != embedding.shape[1]:
            raise ValueError(
                f"Group size mismatch: rewards {rewards.shape[1]} vs embeddings {embedding.shape[1]}"
            )

    def _compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: shape (num_completions, hidden_size)
        Returns:
            sim_matrix: shape (num_completions, num_completions)
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        return embeddings_norm @ embeddings_norm.T

    def _calculate_smi(self) -> torch.Tensor:
        """
        Calculate Submodular Mutual Information for all completions.
        
        SMI(o_i) = sum_{j != i} similarity(o_i, o_j)
        
        Returns:
            smi_scores: shape (batch_size, num_completions)
        """
        batch_size, num_completions, _ = self.embedding.shape
        smi_scores = torch.zeros(
            batch_size, num_completions, 
            device=self.rewards.device, 
            dtype=self.rewards.dtype
        )
        
        for batch_idx in range(batch_size):
            sim_matrix = self._compute_similarity(self.embedding[batch_idx])
            # Sum of row minus diagonal (exclude self-similarity)
            smi_scores[batch_idx] = sim_matrix.sum(dim=-1) - torch.diagonal(sim_matrix)
        
        return smi_scores

    def compute_rewards(self) -> torch.Tensor:
        """
        Apply DRA-GRPO diversity adjustment.
        
        Returns:
            adjusted_rewards: shape (batch_size, num_completions)
        """
        smi_scores = self._calculate_smi()
        return self.rewards / (1.0 + smi_scores + self.epsilon)
