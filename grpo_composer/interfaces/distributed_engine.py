"""
Distributed Engine Interface

Abstract base class for distributed training backends.

Defines:
-------
- `DistributedEngine(ABC)`: Interface for distributed setup

Key methods:
-----------
```python
def initialize(self, config: Dict) -> None:
    '''Initialize distributed backend (FSDP, DeepSpeed, etc).'''
    pass

def wrap_model(self, model: nn.Module) -> nn.Module:
    '''Wrap model for distributed training.'''
    pass

def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
    '''Reduce tensor across all ranks.'''
    pass
```

Implemented by:
--------------
- `runtime/distributed/fsdp.py` - FSDP backend
- `runtime/distributed/deepspeed.py` - DeepSpeed backend
- `runtime/distributed/megatron.py` - Megatron-LM backend

Abstracts away distributed framework differences.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class DistributedEngine(ABC):
    @abstractmethod
    def initialize(self, config: Dict) -> None:
        '''Initialize distributed backend (FSDP, DeepSpeed, etc).'''
        pass

    @abstractmethod
    def wrap_model(self, model:nn.Module) -> nn.Module:
        '''Wrap model for distributed training.'''
        pass

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        '''Reduce tensor across all ranks.'''
        pass
