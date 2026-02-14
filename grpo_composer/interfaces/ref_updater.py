"""
Reference Model Updater Interface

Abstract base class for reference model update strategies.

Defines:
-------
- `RefModelUpdater(ABC)`: Interface for θ_ref update logic

Implemented by:
--------------
- `models/updater/frozen.py` - Never update
- `models/updater/ema.py` - Exponential moving average
- `models/updater/periodic.py` - Hard copy every N steps

Key methods:
-----------
```python
def should_update(self, step: int) -> bool:
    '''Check if ref model should be updated this step.'''
    pass

def update(
    self,
    ref_model: nn.Module,
    policy_model: nn.Module
) -> None:
    '''Update ref model parameters from policy.'''
    pass
```
"""
