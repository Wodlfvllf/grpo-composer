"""
Batch Sampler Interface

Abstract base class for sampling training batches from buffer.

Defines:
-------
- `BatchSampler(ABC)`: Interface for buffer sampling strategies
- `TrainingBatch`: Dataclass for batched training data

Implemented by:
--------------
- `data/sampling/uniform.py` - Random sampling
- `data/sampling/dapo.py` - Filter μ∈{0,1}
- `data/sampling/daro.py` - Difficulty bins
- `data/sampling/pvpo.py` - GT injection
- `data/sampling/xrpo.py` - UCB priority

Key method:
----------
```python
def sample(
    self,
    buffer: Buffer,
    batch_size: int
) -> TrainingBatch:
    '''Sample batch from buffer for training.'''
    pass
```
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import random

class BatchSampler(ABC):
    
    @abstractmethod
    def sample(self, buffer, batch_size)
