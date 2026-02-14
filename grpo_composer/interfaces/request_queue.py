"""
Request Queue Interface

Abstract base class for managing generation request queues.

Defines:
-------
- `RequestQueue(ABC)`: Interface for queueing rollout requests

Key methods:
-----------
```python
def enqueue(self, requests: List[RolloutRequest]) -> None:
    '''Add generation requests to queue.'''
    pass

def dequeue(self, max_batch_size: int) -> List[RolloutRequest]:
    '''Get next batch of requests for generation.'''
    pass

def size(self) -> int:
    '''Current queue size.'''
    pass
```

Implemented by:
--------------
- `rollouts/queue/fifo.py` - First-in-first-out
- `rollouts/queue/priority.py` - Priority-based (XRPO)
- `rollouts/queue/batched.py` - Batch-aware grouping

Used for asynchronous generation and batching optimization.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional

class RequestQueue(ABC):
    @abstractmethod
    def enqueue(self, requests: List[RolloutRequest]) -> None:
        '''Add generation requests to queue.'''
        pass

    @abstractmethod
    def dequeue(self, max_batch_size: int) -> List[RolloutRequest]:
        '''Get next batch of requests for generation.'''
        pass

    @abstractmethod
    def size(self) -> int:
        '''Current queue size.'''
        pass