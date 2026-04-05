from typing import List


class Optimizer:
    """Base class for optimizers."""

    def step(self, layers: List) -> None:
        """Update parameters of the given layers."""
        raise NotImplementedError

    def set_lr(self, lr: float) -> None:
        """Set current learning rate."""
        self.lr = lr


class GradientDescent(Optimizer):
    """
    Vanilla gradient descent optimizer with optional momentum.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        momentum : float
            Momentum factor in [0, 1). 0 disables momentum.
        """
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}

    def step(self, layers: List) -> None:
        for layer in layers:
            if not hasattr(layer, "params") or not hasattr(layer, "grads"):
                continue

            for name, param in layer.params.items():
                grad = layer.grads.get(name)
                if grad is None:
                    continue

                key = (id(layer), name)
                if self.momentum > 0.0:
                    v_prev = self._velocity.get(key, 0.0)
                    v_new = self.momentum * v_prev - self.lr * grad
                    self._velocity[key] = v_new
                    param += v_new
                else:
                    param += -self.lr * grad


class StepLR:
    """
    Step learning-rate scheduler.

    Reduces learning rate by `gamma` every `step_size` epochs.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError("step_size must be > 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr

    def get_lr(self, epoch: int) -> float:
        """
        Return LR to use for the given (1-indexed) epoch.
        """
        step_count = epoch // self.step_size
        return self.base_lr * (self.gamma ** step_count)

    def step(self, epoch: int) -> float:
        """
        Update optimizer LR for current epoch and return it.
        """
        new_lr = self.get_lr(epoch)
        self.optimizer.set_lr(new_lr)
        return new_lr

    def step_epoch_start(self, epoch: int) -> float:
        """Alias for model training loop compatibility."""
        return self.step(epoch)


class ReduceLROnPlateau:
    """
    Reduce learning rate when a monitored metric stops improving.

    Typical usage monitors validation loss.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        min_delta: float = 0.0,
    ):
        if not (0.0 < factor < 1.0):
            raise ValueError("factor must be in (0, 1)")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_lr <= 0.0:
            raise ValueError("min_lr must be > 0")
        if min_delta < 0.0:
            raise ValueError("min_delta must be >= 0")

        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta

        self.best = None
        self.num_bad_epochs = 0

    def step_metric(self, metric: float) -> float:
        """
        Update LR based on the monitored metric and return current LR.
        """
        if self.best is None:
            self.best = metric
            return self.optimizer.lr

        improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            return self.optimizer.lr

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            self.optimizer.set_lr(new_lr)
            self.num_bad_epochs = 0

        return self.optimizer.lr

