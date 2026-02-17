"""
Rollout Worker — Generates G completions per prompt and packs into BufferEntry.

This is the "engine room." Given a list of prompts, it:
    1. Expands each prompt into G generation requests
    2. Calls Generator.generate() → completions + policy_log_probs
    3. Calls RewardEvaluator.compute_rewards() → per-completion rewards
    4. Calls ReferenceModel.get_log_probs() → ref_log_probs (for KL and ratio)
    5. Computes metadata (μ_q, correctness flags, lengths)
    6. Packs everything into List[BufferEntry]

═══════════════════════════════════════════════════════════════════
WHO CALLS THIS:
    rollouts/coordinator.py  →  worker.generate(prompts)
    
WHAT THIS CALLS:
    interfaces/generator.py      →  Generator.generate(requests) → List[RolloutResult]
    interfaces/reward_model.py   →  RewardEvaluator.compute_rewards(prompts, completions)
    interfaces/reference_model.py → ReferenceModel.get_log_probs(token_ids, mask)
═══════════════════════════════════════════════════════════════════

INPUT:
    prompts: List[PromptEntry]  — each has prompt_id, prompt_tokens, etc.
    
OUTPUT:
    List[BufferEntry] — one per prompt, each containing:
        prompt_id         : str
        prompt_tokens     : (T_prompt,)
        completions       : (G, T_completion)
        attention_masks   : (G, T_total)
        policy_log_probs  : (G, T_completion)   ← from Generator (πθ_old)
        ref_log_probs     : (G, T_completion)   ← from ReferenceModel (πref)
        rewards           : (G,)                ← from RewardEvaluator
        completion_lengths: (G,)                ← actual token count per completion
        mean_accuracy     : float               ← μ_q = mean(correctness)
        metadata          : dict                ← variant-specific extras

═══════════════════════════════════════════════════════════════════
VARIANT-SPECIFIC BEHAVIOR:
═══════════════════════════════════════════════════════════════════

Most variants don't change the Worker. The Worker always produces the same
BufferEntry structure. The differences show up in:
    - Which RewardEvaluator is used (binary, frequency-aware, length-dependent, etc.)
    - Which ReferenceModel is used (frozen, EMA, none)

Exceptions where Worker itself changes:
    - Unlikeliness-GRPO: Worker must also compute rank(y_i) by probability
      under πθ_old. This means sorting completions by sum(policy_log_probs)
      and storing the rank in metadata.
    
    - GAPO: Worker must compute group-aware frequency f_v(o) across the G
      completions. This means tracking which completions are identical and
      computing frequency before passing to RewardEvaluator.
    
    - DRA-GRPO: Worker must compute pairwise cosine similarity between
      completion embeddings. This means running an embedding model and
      storing similarity scores in metadata.
    
    - XRPO: Worker must compute per-completion novelty score
      s(y) = (1/|y|) Σ_t log πθ(y_t | x, y_<t)
      This is just mean(policy_log_probs), already available.

For these cases, the Worker should have optional hooks or the metadata
dict should be flexible enough to carry variant-specific data.

═══════════════════════════════════════════════════════════════════
CONSTRUCTOR DEPENDENCIES:
═══════════════════════════════════════════════════════════════════

    generator      : Generator          — HF, vLLM, or TRT-LLM engine
    reward_eval    : RewardEvaluator    — rule-based, learned, or composite
    ref_model      : ReferenceModel     — frozen copy of πref (optional if β=0)
    group_size     : int                — G (number of completions per prompt)
    tokenizer      : Tokenizer          — for decoding completions to text (reward eval needs text)
"""
