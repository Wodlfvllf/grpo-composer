"""
Base Dataset Interface

Defines the abstract interface that all dataset loaders must implement.

Purpose:
-------
Provides a consistent API for loading prompts regardless of the underlying
data source (HuggingFace, JSONL, custom database, etc.).

Role in Training Pipeline:
-------------------------
**Phase: Training Loop Initialization**
- Trainer calls `dataset.sample(batch_size)` to get prompts
- Prompts are passed to RolloutPlanner to create generation requests

Key Responsibilities:
--------------------
1. Define abstract interface for dataset loading
2. Specify required methods: `__len__`, `__getitem__`, `sample`
3. Support both indexed access and random sampling
4. Handle dataset iteration and shuffling

Interface Contract:
------------------
All dataset implementations must provide:
- `__len__() -> int`: Total number of prompts
- `__getitem__(idx: int) -> str`: Get prompt by index
- `sample(batch_size: int) -> List[str]`: Random sample of prompts
- `reset()`: Reset iterator (for epoch-based training)

Classes to Define:
-----------------
1. **Dataset (ABC)**
   - Abstract base class
   - Defines interface contract
   - Provides common utilities (shuffling, batching)

Example Implementation Signature:
---------------------------------
```python
class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        '''Return total number of prompts in dataset.'''
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        '''Get prompt at index idx.'''
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[str]:
        '''
        Sample a batch of prompts.
        
        Args:
            batch_size: Number of prompts to sample
            
        Returns:
            List of prompt strings
        '''
        pass
    
    @abstractmethod
    def reset(self) -> None:
        '''Reset iterator for new epoch.'''
        pass
```

Usage in Training:
-----------------
```python
# In training/trainer.py
prompts = self.dataset.sample(batch_size=4)
# prompts = ["What is 2+2?", "Solve: x^2=16", ...]

# Pass to rollout planner
rollout_requests = self.planner.plan(prompts, ...)
```

Design Decisions:
----------------
- Use ABC to enforce interface contracts
- Return strings (not tokenized) - tokenization happens in generator
- Support both random sampling and sequential iteration
- Keep dataset logic separate from buffer sampling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import random

class Dataset(ABC):
    """
    Abstract base class for all datasets.
    
    All dataset implementations must inherit from this class and implement
    the abstract methods defined below.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        """
        Get a single sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            str: The sample data (e.g., prompt text)
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[str]:
        """
        Sample a batch of samples.
        
        Args:
            batch_size (int): Number of samples to sample
            
        Returns:
            List[str]: List of samples
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the dataset iterator for a new epoch.
        
        This method should reset any internal state that affects iteration order,
        such as shuffle indices or file pointers.
        """
        pass
