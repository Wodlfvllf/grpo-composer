"""
Data Module - Dataset Loading and Prompt Management

This module is responsible for loading and managing training prompts for GRPO.

Role in Training Pipeline:
-------------------------
1. **Startup Phase**: Load dataset specified in config
2. **Training Loop**: Sample batches of prompts to send to RolloutPlanner

Key Responsibilities:
--------------------
- Load prompts from various sources (HuggingFace, JSONL, custom)  
- Provide iterator/sampler interface for training loop
- Handle dataset shuffling, splitting, and streaming
- Support curriculum learning (difficulty-based ordering)

Submodules:
----------
- `datasets/`: Dataset loaders for different formats
- `sampling/`: Batch samplers (from buffer, not from dataset - note the difference)
- `preprocessing.py`: Prompt formatting and templating
- `tokenization.py`: Tokenizer abstraction layer

Note:
----
This module handles **dataset-level** operations (loading prompts).
The `data/sampling/` submodule handles **buffer-level** operations 
(sampling completed rollouts from buffer for training).
"""
