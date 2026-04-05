"""Dropout layer implementation."""

from __future__ import annotations

from ..core.module import Module
from ..core.tensor import Tensor
from ..core.backend import get_backend, randn


class Dropout(Module):
    """Dropout layer for regularization."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Create mask during training
            mask = Tensor(get_backend().xp.random.binomial(1, 1 - self.p, x.data.shape))
            return x * mask / (1 - self.p)
        else:
            # No dropout during evaluation
            return x
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
