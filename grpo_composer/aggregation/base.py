"""
Base Abstract Class for Loss Aggregation in GRPO

Aggregation controls how token-level and group-level losses are combined:
- Token Mean Loss: 1/|o_i| (per-sequence average)
- Group Mean Loss: 1/G (per-group average)

Different papers modify these aggregation strategies:
- Base GRPO: (1/G) * Σ (1/|o_i|) * Σ_t loss_t
- Dr.GRPO: Removes 1/|o_i| (sum instead of mean)
- DAPO: Global token normalization 1/Σ|o_i|
- λ-GRPO: Learnable response-level weights
- DARO: Learnable per-difficulty weights
- TIC-GRPO: Trajectory-level aggregation
- TR-GRPO: Probability-weighted token aggregation
"""
