"""
Diversity Adjusted Reward Calculator (DRA-GRPO)

Implements Diversity-aware Reward Adjustment using Submodular Mutual Information.

Formula:
    R'(q, o_i) = R(q, o_i) * (1 - SMI({o_i}, C \\ {o_i}))

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
    DRA-GRPO: Diversity-Aware Reward Adjustment (paper-faithful)

    Formula:
        R'(q, o_i) = R(q, o_i) / (1 + SMI(o_i, C \ {o_i}))

    where:
        SMI(o_i, C \ {o_i}) = sum_{j != i} cos_sim(e_i, e_j)
    """

    def __init__(
        self,
        rewards: torch.Tensor,           # (B, G)
        embedding: torch.Tensor,         # (B, G, D)
        epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(rewards, **kwargs)
        self.embedding = embedding
        self.epsilon = epsilon

        # Shape validation
        if rewards.ndim != 2:
            raise ValueError(f"Rewards must be 2D, got {rewards.shape}")
        if embedding.ndim != 3:
            raise ValueError(f"Embedding must be 3D, got {embedding.shape}")
        if rewards.shape[:2] != embedding.shape[:2]:
            raise ValueError("Rewards and embeddings must align in (B, G)")

    def _calculate_smi(self) -> torch.Tensor:
        """
        Vectorized SMI computation.

        Returns:
            smi_scores: (B, G)
        """
        # Step 1: normalize embeddings
        emb = F.normalize(self.embedding, p=2, dim=-1)  # (B, G, D)

        # Step 2: similarity matrix
        sim_matrix = torch.matmul(emb, emb.transpose(-1, -2))  # (B, G, G)

        # Step 3: subtract diagonal (self-similarity = 1)
        diag = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)  # (B, G)

        smi_scores = sim_matrix.sum(dim=-1) - diag  # (B, G)

        return smi_scores

    def compute_rewards(self) -> torch.Tensor:
        """
        Apply DRA reward adjustment.

        Returns:
            adjusted_rewards: (B, G)
        """
        smi_scores = self._calculate_smi()

        # ---- Core formula (paper EXACT) ----
        multiplier = 1.0 / (1.0 + smi_scores + self.epsilon)

        return self.rewards * multiplier