"""
HuggingFace Generator

Uses HuggingFace Transformers `.generate()` for text completion.

When to Use:
-----------
- Small models (≤7B) on single GPU
- Prototyping and debugging
- When vLLM not available
- Flexible generation parameters needed

Pros:
----
+ Simple, well-documented
+ Supports all HF models
+ Easy to debug
+ Flexible sampling options

Cons:
----
- Slower than vLLM (no continuous batching)
- Inefficient for large batches
- Memory inefficient for long sequences

Implementation:
--------------
```python
class HFGenerator(Generator):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        '''Load HF model for generation.'''
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def generate(self, requests: List[RolloutRequest]) -> List[RolloutResult]:
        '''Generate using .generate() and extract log-probs from scores.'''
        pass
```

Key Methods:
-----------
- `_extract_log_probs()`: Compute log-probs from generation scores
- `_batch_requests()`: Group requests for batching
- `_postprocess()`: Convert HF outputs to RolloutResult

Performance:
-----------
Typical throughput (7B model, A100):
- Batch size 1: ~20 tokens/sec
- Batch size 8: ~120 tokens/sec
- Batch size 32: ~200 tokens/sec

Compare with vLLM: ~800 tokens/sec with continuous batching
"""
