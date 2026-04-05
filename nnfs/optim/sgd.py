"""Optimizer implementations."""

from __future__ import annotations

from typing import List, Optional

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor import Tensor
from ..core.backend import get_backend, zeros_like, sqrt, maximum, sum


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, parameters: List[Parameter], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.current_step = 0
    
    def zero_grad(self) -> None:
        """Reset gradients of all parameters."""
        for param in self.parameters:
            if param.requires_grad:
                param.zero_grad()
    
    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.lr


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum."""
    
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        gradient_clip: Optional[float] = None
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        
        # Initialize momentum buffers
        self.momentum_buffers = {}
        for param in self.parameters:
            if param.requires_grad:
                self.momentum_buffers[id(param)] = zeros_like(param.data)
    
    def step(self) -> None:
        """Perform SGD optimization step."""
        for param in self.parameters:
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Gradient clipping
            if self.gradient_clip is not None:
                grad_norm = sqrt(sum(grad * grad))
                if grad_norm > self.gradient_clip:
                    grad = grad * (self.gradient_clip / grad_norm)
            
            # Apply momentum
            if self.momentum != 0:
                buf = self.momentum_buffers[id(param)]
                buf = self.momentum * buf + grad
                self.momentum_buffers[id(param)] = buf
                grad = buf
            
            # Update parameters
            param.data = param.data - self.lr * grad
        
        self.current_step += 1


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize Adam state
        self.state = {}
        for param in self.parameters:
            if param.requires_grad:
                self.state[id(param)] = {
                    'step': 0,
                    'exp_avg': zeros_like(param.data),
                    'exp_avg_sq': zeros_like(param.data)
                }
    
    def step(self) -> None:
        """Perform Adam optimization step."""
        for param in self.parameters:
            if not param.requires_grad or param.grad is None:
                continue
            
            grad = param.grad.data
            state = self.state[id(param)]
            
            state['step'] += 1
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first and second moment estimates
            state['exp_avg'] = self.betas[0] * state['exp_avg'] + (1 - self.betas[0]) * grad
            state['exp_avg_sq'] = self.betas[1] * state['exp_avg_sq'] + (1 - self.betas[1]) * (grad * grad)
            
            # Bias correction
            bias_correction1 = 1 - self.betas[0] ** state['step']
            bias_correction2 = 1 - self.betas[1] ** state['step']
            
            # Update parameters
            step_size = self.lr * sqrt(bias_correction2) / bias_correction1
            param.data = param.data - step_size * state['exp_avg'] / (sqrt(state['exp_avg_sq']) + self.eps)
        
        self.current_step += 1
