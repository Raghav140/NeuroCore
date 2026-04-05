"""Activation function implementations."""

from __future__ import annotations

from ..core.module import Module
from ..core.tensor import Tensor


class ReLU(Module):
    """ReLU activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    
    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    """Softmax activation function."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        # Softmax = exp(x - max(x)) / sum(exp(x - max(x)))
        x_max = x.max(axis=self.dim, keepdims=True)
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        sum_exp = exp_x.sum(axis=self.dim, keepdims=True)
        return exp_x / sum_exp
    
    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"
