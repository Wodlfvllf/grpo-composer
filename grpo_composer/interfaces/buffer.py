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

from torch.utils.data import Dataset
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class BufferEntry:
    buffer: list = field(default_factory=list)


class Buffer(ABC):

    @abstractmethod
    def insert(self, entries : List[BufferEntry]) -> None:
        pass

    @abstractmethod
    def get_by_prompt_id(self, prompt_id : str) -> List[BufferEntry]:
        pass

    @abstractmethod
    def get_unique_prompt_ids(self) -> List[str]:
        pass


