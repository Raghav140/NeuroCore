"""Normalization layer implementations."""

from __future__ import annotations

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor import Tensor
from ..core.backend import get_backend, zeros, ones


class BatchNorm1d(Module):
    """Batch Normalization 1D layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = Parameter(ones(num_features))
        self.bias = Parameter(zeros(num_features))
        
        # Running statistics (buffers)
        self.register_buffer('running_mean', zeros(num_features))
        self.register_buffer('running_var', ones(num_features))
        
        # Training flag
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Training mode: use batch statistics
            mean = x.mean(axis=0, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.squeeze()
        else:
            # Evaluation mode: use running statistics
            mean = Tensor(self.running_mean)
            var = Tensor(self.running_var)
        
        # Normalize
        x_normalized = (x - mean) / (var + self.eps).sqrt()
        
        # Scale and shift
        return self.weight * x_normalized + self.bias
    
    def __repr__(self) -> str:
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"
