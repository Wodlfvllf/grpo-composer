"""
Adaptive Planner — Priority-based prompt selection using historical stats.

Used by XRPO for UCB-style exploration-exploitation rollout planning.
Can also support curriculum-based selection for DARO.

═══════════════════════════════════════════════════════════════════
USED BY:
    XRPO   — UCB priority scoring  (Π_q = Δ̂_q + φ_q)
    DARO   — Difficulty-based curriculum (optional, can also use fixed)
═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
XRPO ALGORITHM:
═══════════════════════════════════════════════════════════════════

    For each prompt q with n_q previous rollouts:

    1. Uncertainty (confidence interval):
       h_q(n_q) = t_{1-α/2, n_q-1} * (s_q / √n_q)

    2. Uncertainty reduction from one more rollout:
       Δ̂_q(n_q) = h_q(n_q) - h_q(n_q + 1)

    3. Exploration bonus (UCB-style):
       φ_q(T, n_q) = λ * log(1 + T) / n_q

    4. Priority score:
       Π_q = Δ̂_q(n_q) + φ_q(T, n_q)

    5. Select top-K prompts by Π_q

═══════════════════════════════════════════════════════════════════
INTERNAL STATE (persistent across training steps):
═══════════════════════════════════════════════════════════════════

    self._history = {
        prompt_id: {
            "n_q"         : int,    — number of rollouts so far
            "mean_reward" : float,  — running mean of rewards
            "std_reward"  : float,  — running std of rewards
            "rewards"     : list,   — all rewards seen (for exact stats)
        }
    }

    Updated by coordinator after each generation round.

═══════════════════════════════════════════════════════════════════
CONSTRUCTOR ARGS:
═══════════════════════════════════════════════════════════════════

    alpha           : float = 0.05   — significance level for t-distribution
    lambda_explore  : float = 0.1    — exploration coefficient
    min_rollouts    : int   = 2      — minimum n_q before using stats (else high priority)
"""
