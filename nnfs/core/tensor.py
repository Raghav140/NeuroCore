"""Dynamic autograd tensor engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Set

from .backend import asarray, xp


def _unbroadcast(grad, shape):
    """Reduce broadcasted gradient back to the operand shape."""
    if grad.shape == shape:
        return grad
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


@dataclass(eq=False)
class Tensor:
    """Tensor with dynamic autograd graph."""

    data: object
    requires_grad: bool = False
    _children: Sequence["Tensor"] = ()
    _op: str = ""

    def __post_init__(self):
        self.data = asarray(self.data)
        self.grad = xp().zeros_like(self.data) if self.requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(self._children)

    def __hash__(self):
        return id(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self):
        from .backend import to_numpy

        return to_numpy(self.data)

    def __array__(self, dtype=None):
        arr = self.numpy()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def zero_grad(self):
        if self.requires_grad:
            self.grad = xp().zeros_like(self.data)

    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors.")
            grad = xp().ones_like(self.data)
        grad = asarray(grad)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad - out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data**power, self.requires_grad, (self,), f"pow{power}")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad.T

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    for ax in sorted([a if a >= 0 else a + self.data.ndim for a in axes]):
                        grad = xp().expand_dims(grad, axis=ax)
                self.grad = self.grad + xp().ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __ge__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data >= other_data

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data > other_data

    def __le__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data <= other_data

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data < other_data

    def exp(self):
        out_data = xp().exp(self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out_data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(xp().log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out_data = xp().tanh(self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (1.0 - out_data**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out_data = xp().maximum(0.0, self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out


def tensor(data, requires_grad: bool = False) -> Tensor:
    """Convenience factory."""
    return Tensor(data, requires_grad=requires_grad)
