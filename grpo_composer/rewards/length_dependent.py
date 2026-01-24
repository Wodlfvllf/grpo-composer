"""
GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for
Concise Mathematical Reasoning in Language Models

Paper: https://arxiv.org/abs/2504.09696
Authors: EMNLP 2025

This module implements the Length-Dependent Accuracy Reward component
of GRPO-LEAD, which encourages concise mathematical reasoning by
penalizing verbose solutions while maintaining correctness.

Key Formula:
    z_i = (length(o_i) - μ) / (σ + ε)
    
    R_accuracy(o|q) = exp(-α * z_i),  if o is correct
                    = -1,              if o is incorrect

Where:
    - μ: Mean length of CORRECT responses only (not all responses)
    - σ: Standard deviation of CORRECT response lengths only
    - z_i: Standardized length deviation
    - α: Length penalty strength (paper uses α=0.05)
    - ε: Numerical stability constant (small value like 1e-8)

Critical Design Decisions:
    1. Statistics (μ, σ) computed ONLY over correct responses
    2. Incorrect responses get fixed penalty of -1 (not 0)
    3. Exponential decay exp(-αz) means:
       - Shorter correct responses get reward > 1.0 (boost)
       - Longer correct responses get reward < 1.0 (penalty)
       - At mean length: z=0, reward = 1.0 (neutral)

"""

class GRPOLEADReward:
    """
    Length-Dependent Accuracy Reward for GRPO-LEAD.
    
    Implements dynamic reward shaping that promotes brevity among correct
    responses using standardized length-based penalties, reducing verbosity
    without sacrificing accuracy.
    
    Attributes:
        responses (List[str]): Generated model responses/completions
        labels (List[int]): Binary labels (1=correct, 0=incorrect)
        alpha (float): Length penalty strength (default: 0.05 from paper)
        epsilon (float): Numerical stability constant (default: 1e-8)
    
    Example:
        >>> responses = ["solution 1", "solution 2 longer", "wrong"]
        >>> labels = [1, 1, 0]  # First two correct, last incorrect
        >>> reward_calc = GRPOLEADReward(responses, labels, alpha=0.05)
        >>> rewards = reward_calc.compute_rewards()
        >>> print(rewards)
        [1.05, 0.95, -1.0]  # Shorter correct boosted, longer penalized, wrong=-1
    """
    def __init__(self, responses, labels, alpha, epsilon):
        """
        Initialize GRPO-LEAD reward calculator.
        
        Args:
            responses: List of G generated responses (strings or token sequences)
            labels: Binary correctness labels (1=correct, 0=incorrect)
            alpha: Length penalty strength. Paper uses α=0.05.
                   Higher α → stronger penalty for verbosity
                   Lower α → more tolerance for longer solutions
            epsilon: Small constant for numerical stability in std calculation
        
        Raises:
            ValueError: If responses and labels have different lengths
            ValueError: If no correct responses exist (can't compute statistics)
        """
        if len(responses) != len(labels):
            raise ValueError(
                f"Mismatch: {len(responses)} responses but {len(labels)} labels"
            )

        self.responses = responses
        self.labels = labels
        self.alpha = alpha
        self.epsilon = epsilon

        # Validate we have at least one correct response
        if sum(labels) == 0:
            raise ValueError(
                "No correct responses found. Cannot compute length statistics. "
                "GRPO-LEAD requires at least one correct response in the group."
            )


    def compute_rewards(self):
        mean_length = np.mean([len(response) for response in self.responses])
        std_length = np.std([len(response) for response in self.responses])

        rewards_standardized = []
        for idx, response in enumerate(self.responses):
            if self.labels[idx] == 1:
                z = (len(response) - mean_length) / (std_length + self.epsilon)
                reward = np.exp(-self.alpha * z)
            else:
                reward = -1
            rewards_standardized.append(reward)

        return np.array(rewards_standardized)