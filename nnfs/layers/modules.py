"""Layer and activation modules."""

from __future__ import annotations

from typing import Optional

from ..core.backend import asarray, xp
from ..core.module import Module, Parameter
from ..core.tensor import Tensor


class Dense(Module):
    """Fully connected linear layer."""

    def __init__(self, in_features: int, out_features: int, init: str = "xavier"):
        super().__init__()
        if init == "xavier":
            scale = (2.0 / (in_features + out_features)) ** 0.5
        elif init == "he":
            scale = (2.0 / in_features) ** 0.5
        else:
            raise ValueError("init must be xavier or he")
        self.weight = Parameter(xp().random.randn(in_features, out_features) * scale)
        self.bias = Parameter(xp().zeros((1, out_features)))
        self._x = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        return x @ self.weight + self.bias

    def backward(self, grad):
        # Manual backward kept only for backward compatibility.
        raise RuntimeError("Dense uses autograd Tensor.backward(); call loss.backward().")


class BatchNorm1d(Module):
    """BatchNorm over feature dimension."""

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.gamma = Parameter(xp().ones((1, num_features)))
        self.beta = Parameter(xp().zeros((1, num_features)))
        self.running_mean = xp().zeros((1, num_features))
        self.running_var = xp().ones((1, num_features))
        self.momentum = momentum
        self.eps = eps
        self._cache = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        if self.training:
            mu = x.data.mean(axis=0, keepdims=True)
            var = x.data.var(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            x_hat = (x - Tensor(mu)) * Tensor(1.0 / xp().sqrt(var + self.eps))
            out = self.gamma * x_hat + self.beta
            return out
        x_hat = (x - Tensor(self.running_mean)) * Tensor(1.0 / xp().sqrt(self.running_var + self.eps))
        return self.gamma * x_hat + self.beta

    def backward(self, grad):
        raise RuntimeError("BatchNorm1d uses autograd Tensor.backward(); call loss.backward().")


class Dropout(Module):
    """Dropout regularization layer."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self._mask = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        self._mask = (xp().random.rand(*x.shape) < keep) / keep
        return x * Tensor(self._mask)

    def backward(self, grad):
        raise RuntimeError("Dropout uses autograd Tensor.backward(); call loss.backward().")


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._x = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        return x.relu()

    def backward(self, grad):
        raise RuntimeError("ReLU uses autograd Tensor.backward(); call loss.backward().")


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self._y = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        self._y = Tensor(1.0) / (Tensor(1.0) + (-x).exp())
        return self._y

    def backward(self, grad):
        raise RuntimeError("Sigmoid uses autograd Tensor.backward(); call loss.backward().")


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self._y = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        return x.tanh()

    def backward(self, grad):
        raise RuntimeError("Tanh uses autograd Tensor.backward(); call loss.backward().")


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self._y = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        shifted = x - Tensor(x.data.max(axis=1, keepdims=True))
        ex = shifted.exp()
        return ex / ex.sum(axis=1, keepdims=True)

    def backward(self, grad):
        raise RuntimeError("Softmax uses autograd Tensor.backward(); call loss.backward().")


class Flatten(Module):
    """Flatten input from NCHW/any to (batch, -1)."""

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        raise RuntimeError("Flatten uses autograd Tensor.backward(); call loss.backward().")


class Embedding(Module):
    """Simple embedding lookup layer."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = Parameter(0.01 * xp().random.randn(num_embeddings, embedding_dim))
        self._idx = None

    def forward(self, x):
        self._idx = asarray(x).astype(int)
        out_data = self.weight.data[self._idx]
        out = Tensor(out_data, requires_grad=self.weight.requires_grad, _children=(self.weight,), _op="embedding")

        def _backward():
            if self.weight.requires_grad:
                xp().add.at(self.weight.grad, self._idx, out.grad)

        out._backward = _backward
        return out

    def backward(self, grad):
        raise RuntimeError("Embedding uses autograd Tensor.backward(); call loss.backward().")


class Conv2D(Module):
    """Naive Conv2D layer (N, C, H, W)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        k = kernel_size
        scale = (2.0 / (in_channels * k * k)) ** 0.5
        self.weight = Parameter(xp().random.randn(out_channels, in_channels, k, k) * scale)
        self.bias = Parameter(xp().zeros((out_channels,)))

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        x_data = x.data
        w = self.weight.data
        b = self.bias.data
        n, c, h, ww = x_data.shape
        out_c, _, kh, kw = w.shape
        s, p = self.stride, self.padding
        x_pad = xp().pad(x_data, ((0, 0), (0, 0), (p, p), (p, p))) if p > 0 else x_data
        out_h = (h + 2 * p - kh) // s + 1
        out_w = (ww + 2 * p - kw) // s + 1
        out_data = xp().zeros((n, out_c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                x_slice = x_pad[:, :, i * s : i * s + kh, j * s : j * s + kw]
                out_data[:, :, i, j] = xp().tensordot(x_slice, w, axes=([1, 2, 3], [1, 2, 3])) + b

        out = Tensor(out_data, requires_grad=(x.requires_grad or self.weight.requires_grad or self.bias.requires_grad), _children=(x, self.weight, self.bias), _op="conv2d")

        def _backward():
            if out.grad is None:
                return
            grad_out = out.grad
            if x.requires_grad:
                dx_pad = xp().zeros_like(x_pad)
            if self.weight.requires_grad:
                dw = xp().zeros_like(w)
            if self.bias.requires_grad:
                self.bias.grad += grad_out.sum(axis=(0, 2, 3))
            for i in range(out_h):
                for j in range(out_w):
                    x_slice = x_pad[:, :, i * s : i * s + kh, j * s : j * s + kw]
                    g = grad_out[:, :, i, j]
                    if self.weight.requires_grad:
                        dw += xp().tensordot(g, x_slice, axes=([0], [0]))
                    if x.requires_grad:
                        dx_pad[:, :, i * s : i * s + kh, j * s : j * s + kw] += xp().tensordot(g, w, axes=([1], [0]))
            if self.weight.requires_grad:
                self.weight.grad += dw
            if x.requires_grad:
                x.grad += dx_pad[:, :, p : p + h, p : p + ww] if p > 0 else dx_pad

        out._backward = _backward
        return out

    def backward(self, grad):
        raise RuntimeError("Conv2D uses autograd Tensor.backward(); call loss.backward().")


class MaxPooling2D(Module):
    """Naive max pooling layer (N, C, H, W)."""

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(asarray(x), requires_grad=False)
        xd = x.data
        n, c, h, w = xd.shape
        k, s = self.kernel_size, self.stride
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        out_data = xp().zeros((n, c, out_h, out_w))
        max_idx = {}
        for i in range(out_h):
            for j in range(out_w):
                window = xd[:, :, i * s : i * s + k, j * s : j * s + k]
                flat = window.reshape(n, c, -1)
                idx = flat.argmax(axis=2)
                out_data[:, :, i, j] = flat.max(axis=2)
                max_idx[(i, j)] = idx

        out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="maxpool2d")

        def _backward():
            if x.requires_grad:
                dx = xp().zeros_like(xd)
                for i in range(out_h):
                    for j in range(out_w):
                        idx = max_idx[(i, j)]
                        grad_ij = out.grad[:, :, i, j]
                        for n_i in range(n):
                            for c_i in range(c):
                                pos = int(idx[n_i, c_i])
                                pi, pj = divmod(pos, k)
                                dx[n_i, c_i, i * s + pi, j * s + pj] += grad_ij[n_i, c_i]
                x.grad += dx

        out._backward = _backward
        return out

    def backward(self, grad):
        raise RuntimeError("MaxPooling2D uses autograd Tensor.backward(); call loss.backward().")
