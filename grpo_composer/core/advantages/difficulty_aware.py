"""
GRPO-LEAD: Length-Dependent Reward + Difficulty-Aware Advantage Reweighting

Paper: GRPO-LEAD

Components Changed (from base GRPO):
- Reward: Length-penalized accuracy reward
- Advantage: Difficulty-aware reweighting

Mathematical Form:
    Length-penalized reward:
        z = (|o| - μ) / (σ + ε)                   # Standardized length
        R_acc(o|q) = exp(-α*z)  if correct
                   = -1         if incorrect

    Difficulty proxy:
        ρ_q = (# correct) / (# total)

    Logistic weight:
        w(ρ_q) = A + (B - A) / (1 + exp[k(ρ_q - ρ_0)])

    Difficulty-aware advantage:
        A'_i = Ã_i * w(ρ_q)      if Ã_i > 0
             = Ã_i * w(1 - ρ_q)  if Ã_i ≤ 0
"""

class DifficultyAwareAdvantageFunction(AdvantageFunction):
    def __init__(self, A, B, k, rho_0, epsilon):
        super().__init__()
        self.A = A
        self.B = B
        self.k = k
        self.rho_0 = rho_0
        self.epsilon = epsilon

    def compute_advantages(self, rewards):
        # difficulty proxy
        difficulty_proxy = rewards.sum(dim = -1) / rewards.size(-1)

        # logistic weight
        logistic_weight = self.A + (self.B - self.A) / (1 + torch.exp(self.k * (difficulty_proxy - self.rho_0)))

        # base advantage
        base_advantage = (rewards - rewards.mean(dim = -1, keepdim = True)) / (rewards.std(dim = -1, keepdim = True) + self.epsilon)

        # difficulty-aware advantage
        difficulty_aware_advantage = torch.where(base_advantage > 0, base_advantage * logistic_weight, base_advantage * (1 - logistic_weight))

        return difficulty_aware_advantage

        