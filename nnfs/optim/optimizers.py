"""Optimizers."""

from __future__ import annotations

from typing import Dict, Iterable

from ..core.module import Parameter


class SGD:
    """Stochastic gradient descent with optional momentum and clipping."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        grad_clip_norm: float | None = None,
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.grad_clip_norm = grad_clip_norm
        self._vel: Dict[int, object] = {}

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def _clip_grad(self) -> None:
        if self.grad_clip_norm is None:
            return
        total_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            total_sq += float((p.grad**2).sum())
        norm = total_sq**0.5
        if norm <= self.grad_clip_norm or norm == 0.0:
            return
        scale = self.grad_clip_norm / (norm + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            p.grad *= scale

    def step(self) -> None:
        self._clip_grad()
        for p in self.params:
            if p.grad is None:
                continue
            key = id(p)
            if self.momentum > 0.0:
                v_prev = self._vel.get(key, 0.0)
                v_new = self.momentum * v_prev - self.lr * p.grad
                self._vel[key] = v_new
                p.data += v_new
            else:
                p.data -= self.lr * p.grad
