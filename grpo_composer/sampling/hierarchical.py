"""
XRPO: Hierarchical Rollout Planning with Exploration-Exploitation

Paper: XRPO

Components Changed (from base GRPO):
- Prioritizes prompts based on uncertainty + exploration bonus
- ICL seeding for zero-reward prompts

Mathematical Form:
    Uncertainty (confidence interval):
        h_q(n_q) = t_{1-α/2, n_q-1} * s_q / √n_q

    Uncertainty reduction:
        Δ̂_q(n_q) = h_q(n_q) - h_q(n_q + 1)

    Exploration bonus:
        φ_q(T, n_q) = λ * √(log(1+T) / n_q)

    Priority score:
        Π_q = Δ̂_q(n_q) + φ_q(T, n_q)

    Where:
        n_q = number of rollouts for prompt q
        T = total training steps
        s_q = sample std of rewards for q

Special Handling:
    Zero-reward prompts → ICL seeding (few-shot from success corpus)

Effect:
    Balances exploration (try new prompts) vs exploitation (improve on known).
    Adaptive sampling based on learning progress.
"""
