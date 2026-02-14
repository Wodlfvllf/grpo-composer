"""
Base Generator Interface

Defines abstract interface for text generation backends.

Role in Pipeline:
----------------
**Phase 2: Rollout Generation**
- Receives RolloutRequest[] from RolloutPlanner/Queue
- Executes generation using specific backend (HF, vLLM, TensorRT)
- Returns RolloutResult[] with completions + log-probs

Interface Contract:
------------------
```python
class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        requests: List[RolloutRequest]
    ) -> List[RolloutResult]:
        '''Execute generation and return results with log-probs.'''
        pass
```

Data Structures:
---------------
```python
@dataclass
class RolloutRequest:
    prompt_id: str
    prompt_text: str
    num_completions: int      # G
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    top_k: int = -1

@dataclass
class RolloutResult:
    rollout_id: str
    prompt_id: str
    completion_text: str
    token_ids: torch.Tensor       # (T,)
    log_probs: torch.Tensor       # (T,) - policy log-probs
    attention_mask: torch.Tensor  # (T,)
    metadata: Dict = field(default_factory=dict)
```

Implementations:
---------------
- `hf.py`: HuggingFace .generate() - Simple, flexible
- `vllm.py`: vLLM - Fast continuous batching
- `trtllm.py`: TensorRT-LLM - Production optimized

Key Requirements:
----------------
1. Must return log-probs for each token
2. Handle batching efficiently
3. Support temperature/top-p/top-k sampling
4. Return attention masks for padding
"""
