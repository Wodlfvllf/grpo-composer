"""Concrete :class:`FlowPlugin` implementations for composer flows.

Each plugin owns the runtime behavior previously implemented as either a
monkey-patch on a veRL method or an ``if/elif`` branch inside
``ComposerRayPPOTrainer.fit``.
"""

from .info_grpo import InfoGRPOFlowPlugin

__all__ = ["InfoGRPOFlowPlugin"]
