"""
PVPO: Ground Truth Trajectory Injection

Paper: PVPO

Components Changed (from base GRPO):
- For prompts with accuracy=0, inject ground truth trajectory
- Uses larger LLM to generate GT for zero-accuracy samples

Sampling Strategy:
    1. Exclude: Samples with mean accuracy = 1 (trivial, no learning)
    2. Retain: Samples with 0 < μ_q < 1 (nonzero advantage)
    3. Special handling for accuracy = 0:
       - Use larger LLM to generate GT trajectories
       - Cache GT trajectories
       - Replace one rollout with cached GT trajectory

Effect:
    Provides learning signal even for very hard prompts.
    Bootstraps learning from stronger model.
"""
