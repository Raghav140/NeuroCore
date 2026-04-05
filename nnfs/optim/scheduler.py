"""Learning rate scheduler implementations."""

from __future__ import annotations

from typing import List, Optional

from ..core.tensor import Tensor


class _LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.step()
    
    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        for param_group in [self.optimizer]:
            param_group.lr = self.get_lr()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        raise NotImplementedError


class StepLR(_LRScheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        """Calculate learning rate for current epoch."""
        return self.optimizer.lr * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(_LRScheduler):
    """Exponential learning rate scheduler."""
    
    def __init__(self, optimizer, gamma: float = 0.95, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        """Calculate learning rate for current epoch."""
        return self.optimizer.lr * (self.gamma ** self.last_epoch)


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0,
        verbose: bool = False
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode {mode} is unknown!")
        
        self._init_is_better(mode, threshold)
        self._reset()
    
    def _init_is_better(self, mode: str, threshold: float) -> None:
        if mode == 'min':
            self.is_better = lambda a, best: a < best - threshold
            self.monitor_op = lambda a, b: a < b
        else:
            self.is_better = lambda a, best: a > best + threshold
            self.monitor_op = lambda a, b: a > b
    
    def _reset(self) -> None:
        """Reset scheduler state."""
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def step(self, metrics: float) -> None:
        """Update learning rate based on metrics."""
        current = float(metrics)
        
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            old_lr = self.optimizer.lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if old_lr - new_lr > 1e-8:
                self.optimizer.lr = new_lr
                if self.verbose:
                    print(f'ReduceLROnPlateau reducing learning rate to {new_lr}')
                
                self.cooldown_counter = self.patience
                self.num_bad_epochs = 0
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
