"""
Planner Base — Abstract base class for prompt selection strategies.

The Planner answers: "Which prompts should we generate completions for?"
This happens BEFORE generation, not after.

═══════════════════════════════════════════════════════════════════
WHO CALLS THIS:
    rollouts/coordinator.py  →  planner.select(dataset, N)
    
WHAT THIS RETURNS:
    List[PromptEntry] — N prompts selected from the dataset
═══════════════════════════════════════════════════════════════════

IMPLEMENTATIONS:
    fixed.py      — Sequential iteration (standard GRPO, Dr.GRPO, DAPO, etc.)
    adaptive.py   — Priority-based selection (XRPO UCB, DARO curriculum)

═══════════════════════════════════════════════════════════════════
KEY DISTINCTION: Planner vs Sampling
═══════════════════════════════════════════════════════════════════

    Planner  = "Which prompts to GENERATE for"    (before generation)
    
    The old data/sampling/ was "which prompts to TRAIN on" (after generation),
    but that responsibility is now absorbed by the Coordinator's loop logic.

═══════════════════════════════════════════════════════════════════
INTERFACE:
═══════════════════════════════════════════════════════════════════

    class Planner(ABC):
        def select(self, dataset, N, **kwargs) -> List[PromptEntry]:
            '''Select N prompts from dataset for rollout generation.'''
        
        def reset(self):
            '''Reset internal state (e.g., cursor position for fixed planner).'''
"""
