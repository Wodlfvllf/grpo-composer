"""
Buffer Interface

Abstract base class for rollout storage and retrieval.

Defines:
-------
- `Buffer(ABC)`: Interface for rollout buffer
- `BufferEntry`: Dataclass for single rollout

Key methods:
-----------
```python
def insert(self, entries: List[BufferEntry]) -> None:
    '''Add rollouts to buffer.'''
    pass

def get_by_prompt_id(self, prompt_id: str) -> List[BufferEntry]:
    '''Retrieve all rollouts for a prompt.'''
    pass

def get_unique_prompt_ids(self) -> List[str]:
    '''Get all unique prompt IDs in buffer.'''
    pass
```

Implemented by:
--------------
- `rollouts/buffer/in_memory.py` - Local dict-based
- `rollouts/buffer/redis.py` - Distributed Redis
- `rollouts/buffer/disk.py` - Persistent storage
"""
import torch
from torch.utils.data import Dataset
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class BufferEntry:
    completions       : torch.Tensor #(G, T_completion)
    policy_log_probs  : torch.Tensor #(G, T_completion)
    ref_log_probs     : torch.Tensor #(G, T_completion)
    rewards           : torch.Tensor #(G,)
    # completion_lengths: torch.Tensor #(G,)
    # mean_accuracy     : float   #(μ_q)
    # metadata          : dict


class Buffer(ABC):
    @abstractmethod
    def insert(self, entries: List[BufferEntry]) -> None:
        '''Add new rollout entries to the buffer.'''
    @abstractmethod
    def get_all(self) -> List[BufferEntry]:
        '''Return all entries (no filtering).'''
    @abstractmethod
    def clear(self) -> None:
        '''Remove all entries. Called after each on-policy training step.'''
    @abstractmethod
    def current_size(self) -> int:
        '''Number of entries currently stored.'''


