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

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass, field

@dataclass
class RolloutRequest:
    prompt : torch.Tensor
    attention_mask : torch.Tensor   
    params : dict = field(default_factory = dict)

@dataclass
class RolloutResult:
    completion : torch.Tensor
    log_probs : torch.Tensor

class Generator(ABC):
    @abstractmethod
    def generate(self, requests : List[RolloutRequest]) -> List[RolloutResult]:
        pass

