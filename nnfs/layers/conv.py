"""Convolutional and pooling layer implementations."""

from __future__ import annotations

from typing import Optional, Tuple

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor import Tensor
from ..core.backend import get_backend, randn, zeros


class Conv2D(Module):
    """2D Convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Initialize weights
        bound = (6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size)) ** 0.5
        self.weight = Parameter(randn(out_channels, in_channels, kernel_size, kernel_size) * bound)
        
        if bias:
            self.bias = Parameter(zeros(out_channels))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D convolution."""
        # Input shape: (batch_size, in_channels, height, width)
        batch_size, in_channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            # Simple zero padding
            padded_data = get_backend().xp.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
            x_padded = Tensor(padded_data)
        else:
            x_padded = x
        
        # Initialize output
        output_data = get_backend().xp.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Perform convolution (simplified implementation)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # Extract patch
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = x_padded.data[b, :, h_start:h_end, w_start:w_end]
                        
                        # Convolution operation
                        conv_sum = get_backend().xp.sum(patch * self.weight.data[oc])
                        
                        if self.bias is not None:
                            conv_sum += self.bias.data[oc]
                        
                        output_data[b, oc, oh, ow] = conv_sum
        
        return Tensor(output_data)
    
    def __repr__(self) -> str:
        return (f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class MaxPooling2D(Module):
    """2D Max Pooling layer."""
    
    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for 2D max pooling."""
        batch_size, channels, in_h, in_w = x.shape()
        
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


class Flatten(Module):
    """Flatten layer for transitioning between conv and dense layers."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Flatten all dimensions except batch."""
        batch_size = x.shape[0]
        return x.reshape((batch_size, -1))
    
    def __repr__(self) -> str:
        return "Flatten()"
