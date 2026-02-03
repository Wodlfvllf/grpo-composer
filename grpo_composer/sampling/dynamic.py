"""
DAPO: Dynamic Sampling / Oversampling - Filter Uninformative Prompts

Paper: DAPO

Components Changed (from base GRPO):
- Filters prompts where ALL responses have same outcome
- μ=0 (all wrong) or μ=1 (all correct) → zero gradient signal

Mathematical Form:
    Filter indicator:
        I_OS(q) = I[0 < μ_q < 1]

    Where:
        μ_q = (# correct responses) / G

    Include prompt only if:
        0 < μ_q < 1   (at least one correct AND one incorrect)

Rationale:
    Zero-variance prompts provide no gradient signal.
    Skip them to focus compute on informative prompts.
"""
