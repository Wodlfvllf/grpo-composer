"""
Reference Model Interface

Abstract base class for reference model implementations.

Defines:
-------
- `ReferenceModel(ABC)`: Interface for computing ref log-probs

Implemented by:
--------------
- `inference/ref_model/frozen.py` - Local in-memory
- `inference/ref_model/remote.py` - Remote server

Key method:
----------
```python
def get_log_probs(
    self,
    token_ids: torch.Tensor,      # (B, G, T)
    attention_mask: torch.Tensor   # (B, G, T)
) -> torch.Tensor:                 # (B, G, T)
    '''Compute reference log-probabilities for KL regularization.'''
    pass
```
"""
