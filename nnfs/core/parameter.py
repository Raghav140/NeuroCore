"""Parameter class for model parameters."""

from __future__ import annotations

from .tensor import Tensor


class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""
    
    def __init__(self, data: any, requires_grad: bool = True):
        super().__init__(data, requires_grad=True)
    
    def __repr__(self) -> str:
        return f"Parameter(data={self.data}, requires_grad={self.requires_grad})"
