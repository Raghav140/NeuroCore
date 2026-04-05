"""Dense (fully connected) layer implementation."""

from __future__ import annotations

from typing import Optional, Tuple

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor import Tensor
from ..core.backend import get_backend, randn, zeros


class Dense(Module):
    """Dense (fully connected) layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weights using Xavier/Glorot initialization
        bound = (6 / (in_features + out_features)) ** 0.5
        self.weight = Parameter(randn(in_features, out_features) * bound)
        
        if bias:
            self.bias = Parameter(zeros((1, out_features)))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def __repr__(self) -> str:
        return f"Dense(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
