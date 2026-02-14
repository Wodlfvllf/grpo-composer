"""
Inference Engines - Text Generation Backends

This module provides different backends for executing text generation during rollout.

Role in Training Pipeline:
-------------------------
**Phase 2: Rollout Generation**
1. Receive generation requests from RolloutPlanner/Queue
2. Execute parallel generation for G completions per prompt
3. Return completions with log-probabilities

Available Engines:
-----------------
| Engine | Speed | Use Case | Setup |
|--------|-------|----------|-------|
| **HF** | 1x | Prototyping, ≤7B | Easy |
| **vLLM** | 4-10x | Production, 7B-70B | Medium |
| **TensorRT-LLM** | 8-15x | Max performance, quantization | Hard |

Key Requirement:
---------------
All engines must return **log-probabilities** for each generated token.
This is critical for computing the GRPO loss.

Quick Start:
-----------
```python
# Simple HF generator
from grpo_composer.inference.engines import HFGenerator

generator = HFGenerator(model_name="meta-llama/Llama-2-7b-hf")
results = generator.generate(requests)

# Fast vLLM generator
from grpo_composer.inference.engines import VLLMGenerator

generator = VLLMGenerator(
    model_name="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=4
)
results = generator.generate(requests)
```

See individual files for detailed documentation.
"""
