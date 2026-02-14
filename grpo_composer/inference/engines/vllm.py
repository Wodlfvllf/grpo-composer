"""
vLLM Generator

Uses vLLM for high-throughput text generation with continuous batching.

When to Use:
-----------
- Production deployments
- Large models (13B+)
- High throughput requirements
- Multi-GPU inference

Pros:
----
+ 10-30x faster than HF for large batches
+ Continuous batching (no padding waste)
+ PagedAttention (memory efficient)
+ Tensor parallelism built-in
+ Prefix caching support

Cons:
----
- Less flexible than HF
- Requires separate installation
- Limited model support (popular models only)
- More complex setup

Implementation:
--------------
```python
class VLLMGenerator(Generator):
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,
        gpu_memory_utilization: float = 0.9
    ):
        '''Initialize vLLM engine.'''
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization
        )
    
    def generate(self, requests: List[RolloutRequest]) -> List[RolloutResult]:
        '''Use vLLM batched generation with log-probs.'''
        pass
```

Key Features:
------------
- **Continuous Batching**: Requests dynamically added/removed
- **PagedAttention**: KV cache in paged blocks
- **Prefix Caching**: Reuse common prompt prefixes
- **Tensor Parallelism**: Split model across GPUs

Performance:
-----------
Typical throughput (13B model, 4xA100):
- vLLM: ~800-1200 tokens/sec
- HF: ~200-300 tokens/sec
- **4x speedup** with continuous batching

Configuration Tips:
------------------
```yaml
vllm:
  tensor_parallel_size: 4       # GPUs per model
  max_num_seqs: 256             # Max concurrent sequences
  gpu_memory_utilization: 0.9   # GPU memory fraction
  enable_prefix_caching: true   # Cache prompt prefixes
```

Installation:
------------
```bash
pip install vllm
# Requires CUDA 11.8+ and compatible GPU (A100, H100)
```
"""
