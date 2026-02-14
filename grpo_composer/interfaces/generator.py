"""
Generator Interface

Abstract base class for text generation backends.

Defines:
-------
- `Generator(ABC)`: Interface for all generation engines
- `RolloutRequest`: Input dataclass (prompt, params)
- `RolloutResult`: Output dataclass (completion, log-probs)

Implemented by:
--------------
- `inference/engines/hf.py`
- `inference/engines/vllm.py`
- `inference/engines/trtllm.py`

Key method:
----------
```python
def generate(
    self,
    requests: List[RolloutRequest]
) -> List[RolloutResult]:
    '''Execute generation and return completions with log-probs.'''
    pass
```
"""
