"""Pooling layer implementations."""

from __future__ import annotations

from typing import Optional

from ..core.module import Module
from ..core.tensor import Tensor
from ..core.backend import get_backend


class MaxPooling2D(Module):
    """2D Max Pooling layer."""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D max pooling."""
        batch_size, channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output_data = get_backend().xp.zeros((batch_size, channels, out_h, out_w))
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = x.data[b, c, h_start:h_end, w_start:w_end]
                        output_data[b, c, oh, ow] = get_backend().xp.max(patch)
        
        return Tensor(output_data)
    
    def __repr__(self) -> str:
        return f"MaxPooling2D(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPooling2D(Module):
    """2D Average Pooling layer."""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D average pooling."""
        batch_size, channels, in_h, in_w = x.shape()
        
        # Calculate output dimensions
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output_data = get_backend().xp.zeros((batch_size, channels, out_h, out_w))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = x.data[b, c, h_start:h_end, w_start:w_end]
                        output_data[b, c, oh, ow] = get_backend().xp.mean(patch)
        
        return Tensor(output_data)
    
    def __repr__(self) -> str:
        return f"AvgPooling2D(kernel_size={self.kernel_size}, stride={self.stride})"
