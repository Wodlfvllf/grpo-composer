"""
Config Interface

Abstract base class for configuration management.

Defines:
-------
- `Config(ABC)`: Interface for config objects
- Validation, loading, and merging configs

Key methods:
-----------
```python
def validate(self) -> None:
    '''Validate config parameters.'''
    pass

def to_dict(self) -> Dict:
    '''Serialize config to dict.'''
    pass

@classmethod
def from_yaml(cls, path: str) -> 'Config':
    '''Load config from YAML file.'''
    pass
```

Implemented by:
--------------
- `config/schema.py` - Main GRPOConfig dataclass
- Custom configs for specific components

Used throughout codebase for type-safe configuration.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Config(ABC):

    @abstractmethod
    def validate(self):
        '''Validate config parameters.'''
        pass

    @abstractmethod
    def to_dict(self):
        '''Serialize config to dict.'''
        pass

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
