# Unified GRPO Framework v4

## Overview
This repository contains a unified mathematical framework for six Group Relative Policy Optimization (GRPO) variants. The framework consolidates these methods into a single objective function with configurable hyperparameters, allowing for the recovery of individual methods as well as the creation of hybrid configurations.

## Supported Methods
The framework unifies the following variants:
1.  **GRPO** (DeepSeekMath, 2024)
2.  **Dr. GRPO** (Bias-Free, 2025)
3.  **DAPO** (ByteDance, 2025)
4.  **DARO** (Difficulty-Aware, 2025)
5.  **$\lambda$-GRPO** (Token Preferences, 2025)
6.  **DRA-GRPO** (Diversity-Aware, 2025)

## Unified Objective
The master objective function is defined as:

$$
\mathcal{J}_{\text{Unified}}(\theta, \{w_\mu\}, \lambda) = \sum_{\mu \in \mathcal{M}} w_\mu^{\text{eff}} \cdot \mathbb{E}_{q: \mu_q = \mu} \left[ \mathbb{I}_{\text{OS}}(q) \cdot \frac{1}{\Omega_\mu} \sum_{i=1}^{G} f_\lambda(o_i) \cdot w_i^{\text{len}} \sum_{t=1}^{|o_i|} \left( \mathcal{L}_{\text{clip}}^{(i,t)} - \beta \cdot D_{\text{KL}}^{(i,t)} \right) \right] + \mathcal{L}_{\text{reg}}
$$

## Configuration
The framework is controlled by 19 hyperparameters, including:
-   **Clipping**: `epsilon_low`, `epsilon_high`
-   **Normalization**: `std_normalize`, `length_norm`, `group_norm_type`
-   **Regularization**: `beta` (KL penalty)
-   **Sampling**: `oversampling`, `group_size`
-   **Weighting**: `difficulty_weighting`, `lambda_weighting`, `diversity_weighting`

Refer to the mathematical documentation for detailed derivations and recovery proofs.
