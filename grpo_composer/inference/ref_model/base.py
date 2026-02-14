"""
Base Reference Model Interface

Defines how to compute reference log-probabilities during training.

Purpose:
-------
Provide log-probs from reference policy π_ref for KL regularization.

Interface:
---------
```python
class ReferenceModel(ABC):
    @abstractmethod
    def get_log_probs(
        self,
        token_ids: torch.Tensor,      # (B, G, T)
        attention_mask: torch.Tensor   # (B, G, T)
    ) -> torch.Tensor:                 # (B, G, T)
        '''Compute log P(token | prefix) for each token.'''
        pass
```

Used in training for KL regularization:
```python
kl_loss = (policy_log_probs - ref_log_probs) * mask
```
"""
