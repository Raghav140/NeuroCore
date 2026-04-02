"""Learning rate schedulers."""

from __future__ import annotations


class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr

    def step_epoch_start(self, epoch: int) -> float:
        step_count = epoch // self.step_size
        lr = self.base_lr * (self.gamma**step_count)
        self.optimizer.set_lr(lr)
        return lr


class ReduceLROnPlateau:
    def __init__(self, optimizer, factor: float = 0.5, patience: int = 10, min_lr: float = 1e-6, min_delta: float = 0.0):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step_metric(self, metric: float) -> float:
        if self.best is None or metric < (self.best - self.min_delta):
            self.best = metric
            self.bad_epochs = 0
            return self.optimizer.lr
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.optimizer.set_lr(max(self.optimizer.lr * self.factor, self.min_lr))
            self.bad_epochs = 0
        return self.optimizer.lr
