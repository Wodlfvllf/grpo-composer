
"""
RankGRPO: Group Relative Policy Optimization with Intra-Group Preference Ranking

This paper implements RankGRPO, an extension of GRPO that replaces sparse or
unstable scalar rewards with intra-group relative ranking to compute stable
advantages for policy optimization.

Given a prompt p, the policy πθ samples a group of G responses:
    {o₁, o₂, ..., o_G}

Each response is evaluated by:
1. A rule-based verifier producing binary correctness:
       sᵢ_rule ∈ {0, 1}
2. A Ranking Reward Model (RRM) producing a relative rank within the group:
       rᵢ ∈ {1, ..., G}   (1 = best)

The rank and rule reward are combined into a rank-aware reward sᵢ_rank using
one of the following mappings:

Ranking as Weight:
    sᵢ_rank = exp(τ · (1 − rᵢ)) · sᵢ_rule

Ranking as Supplement:
    sᵢ_rank = sᵢ_rule + τ · tanh( (r_max / rᵢ) − 1 )

Ranking as Reward:
    sᵢ_rank = (r_max − rᵢ) / (r_max − 1)

Group-relative advantage is then computed as:
    Aᵢ = (sᵢ_rank − mean({sⱼ_rank})) / F_norm

where F_norm is either 1 or std({sⱼ_rank}).

To prevent rank-induced misalignment with correctness, advantages are clipped:
    Aᵢ_clip =
        max(Aᵢ, ξ⁻)  if sᵢ_rule = 1
        min(Aᵢ, ξ⁺)  if sᵢ_rule = 0

The policy is optimized using the standard GRPO clipped objective:
    J(θ) = E[min(ρᵢ,t(θ) · Aᵢ_clip,
                 clip(ρᵢ,t(θ), 1−ε, 1+ε) · Aᵢ_clip)]

where:
    ρᵢ,t(θ) = πθ(oᵢ,t | sᵢ,t) / πθ_old(oᵢ,t | sᵢ,t)

RankGRPO preserves GRPO’s group-relative optimization while preventing
vanishing gradients caused by identical rewards and instability from
absolute scalar reward magnitudes.

"""

import torch


class RankEnhancedRewardCalculator:
    """
    Computes rank-enhanced rewards for RankGRPO.
    
    Combines binary correctness with intra-group ranking to produce
    stable, informative rewards even when all responses are correct/incorrect.
    """
    
    def __init__(self, 
                 tau: float = 0.1,
                 ranking_method: str = "weight"):
        """
        Args:
            tau: Temperature parameter for rank-based reward scaling
            ranking_method: "weight", "supplement", or "reward"
        """
        self.tau = tau
        self.ranking_method = ranking_method
    
    def compute_rewards(self, rewards: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute rank-enhanced rewards.
        
        Args:
            rewards: (B, G) binary correctness (0 = incorrect, 1 = correct)
            log_probs: (B, G) log probabilities for ranking
            
        Returns:
            s_rank: (B, G) rank-enhanced rewards
        """
        B, G = rewards.shape

        # Rank by probability: higher prob = rank 1 (best)
        # argsort(-log_probs) gives indices sorted by descending prob
        # argsort again gives the rank of each element
        ranks_0indexed = torch.argsort(
            torch.argsort(-log_probs, dim=1),
            dim=1
        )
        # Convert to 1-indexed as per paper: r_i ∈ {1, ..., G}
        ranks = ranks_0indexed + 1  # Now 1 = most likely, G = least likely

        if self.ranking_method == "weight":
            # sᵢ_rank = exp(τ · (1 − rᵢ)) · sᵢ_rule
            s_rank = rewards * torch.exp(self.tau * (1 - ranks))
            
        elif self.ranking_method == "supplement":
            # sᵢ_rank = sᵢ_rule + τ · tanh(r_max / rᵢ − 1)
            s_rank = rewards + self.tau * torch.tanh((G / ranks) - 1)
            
        elif self.ranking_method == "reward":
            # sᵢ_rank = (r_max − rᵢ) / (r_max − 1)
            s_rank = (G - ranks).float() / (G - 1)
            
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")

        return s_rank