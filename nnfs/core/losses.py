"""Loss functions."""

from __future__ import annotations

from ..core.backend import xp
from .tensor import Tensor


class Loss:
    """Base loss class."""

    def forward(self, y_pred, y_true) -> float:
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, Tensor):
            y_true = y_true.reshape(y_pred.shape)
            return float(xp().mean((y_pred - y_true) ** 2))
        yt = Tensor(y_true.reshape(y_pred.shape), requires_grad=False)
        return ((y_pred - yt) ** 2).mean()

    def backward(self, y_pred, y_true):
        yp = y_pred.data if isinstance(y_pred, Tensor) else y_pred
        y_true = y_true.reshape(yp.shape)
        return 2.0 * (yp - y_true)


class BCELoss(Loss):
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        if not isinstance(y_pred, Tensor):
            y_true = y_true.reshape(y_pred.shape)
            yp = xp().clip(y_pred, self.eps, 1 - self.eps)
            return float(xp().mean(-(y_true * xp().log(yp) + (1 - y_true) * xp().log(1 - yp))))

        yt = Tensor(y_true.reshape(y_pred.shape), requires_grad=False)
        yp = y_pred + self.eps
        one = Tensor(1.0)
        return (-(yt * yp.log() + (one - yt) * (one - y_pred + self.eps).log())).mean()

    def backward(self, y_pred, y_true):
        yp = y_pred.data if isinstance(y_pred, Tensor) else y_pred
        y_true = y_true.reshape(yp.shape)
        yp = xp().clip(yp, self.eps, 1.0 - self.eps)
        return -(y_true / yp) + ((1.0 - y_true) / (1.0 - yp))


class CrossEntropyLoss(Loss):
    """Categorical cross entropy for probabilities."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        if isinstance(y_pred, Tensor):
            yp_data = xp().clip(y_pred.data, self.eps, 1 - self.eps)
        else:
            yp_data = xp().clip(y_pred, self.eps, 1 - self.eps)
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_idx = y_true.astype(int).reshape(-1)
            p = yp_data[xp().arange(yp_data.shape[0]), y_idx]
            return float(-xp().mean(xp().log(p)))
        p = xp().sum(y_true * yp_data, axis=1)
        return float(-xp().mean(xp().log(p)))

    def backward(self, y_pred, y_true):
        yp = y_pred.data if isinstance(y_pred, Tensor) else y_pred
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            oh = xp().zeros_like(yp)
            idx = y_true.astype(int).reshape(-1)
            oh[xp().arange(yp.shape[0]), idx] = 1.0
            y_true = oh
        return yp - y_true
