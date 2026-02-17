"""
Rollout Scheduler — Controls WHEN and HOW OFTEN to run rollouts.

═══════════════════════════════════════════════════════════════════
NOTE: This may be merged into coordinator.py if it stays simple.
═══════════════════════════════════════════════════════════════════

PURPOSE:
    Decides whether to run fresh rollouts at a given training step.
    In pure on-policy GRPO: always yes (every step).
    In off-policy/replay setups: only when buffer is stale or depleted.

WHEN IS THIS USEFUL:
    - PVPO replay: Can train on cached rollouts for K steps before refreshing.
    - XRPO: May want to generate extra rollouts for high-priority prompts
      without re-generating for all prompts every step.
    - Efficiency: Generation is ~80% of compute. Skipping generation on
      some steps saves significant wallclock time.

═══════════════════════════════════════════════════════════════════
INTERFACE:
═══════════════════════════════════════════════════════════════════

    class RolloutScheduler:
        def should_generate(self, step, buffer_age, buffer_size) -> bool:
            '''Should we run new rollouts this step?'''
        
        def num_prompts_needed(self, batch_size, valid_in_buffer) -> int:
            '''How many new prompts to generate for?'''

═══════════════════════════════════════════════════════════════════
FOR NOW:
    Keep it simple — always generate. Implement scheduling later
    when you add PVPO replay or want to optimize throughput.
"""
