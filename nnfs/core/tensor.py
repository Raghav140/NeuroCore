"""Tensor implementation with automatic differentiation."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

from .backend import array as backend_array, sum as _sum, zeros_like, get_backend, maximum, minimum, ones_like, reshape


class Tensor:
    """Tensor class with automatic differentiation support."""
    
    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
        _backward: Callable[[], None] | None = None
    ):
        self.data = backend_array(data)
        self.requires_grad = requires_grad
        self._backward = _backward
        self._prev = set(_children)
        self._op = _op
        
        # Gradient storage
        self.grad = None
        if requires_grad:
            self.grad = Tensor(zeros_like(self.data))
    
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    # Utility methods
    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        if self.grad is not None:
            self.grad.data.fill(0)
    
    def backward(self) -> None:
        """Compute gradients using backpropagation."""
        if not self.requires_grad:
            return
        
        # Build topological order
        topo: List[Tensor] = []
        visited = set()
        
        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output as 1
        if self.grad is None:
            self.grad = Tensor(ones_like(self.data))
        else:
            # Set to 1 if it exists but is zero
            self.grad.data = ones_like(self.data)
        
        # Go backward through topological order
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()
    
    def item(self) -> float:
        """Get scalar value as Python float."""
        return float(get_backend().as_numpy(self.data).item())
    
    def numpy(self) -> Any:
        """Convert to NumPy array."""
        return get_backend().as_numpy(self.data)
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
    
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.data.ndim
    
    # Arithmetic operations
    def __add__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += out.grad.data
            if other.requires_grad:
                if other.grad is not None:
                    # Handle broadcasting: sum over axes that were broadcasted
                    grad_to_add = out.grad.data
                    # If self has more dimensions than other, sum over the extra axes
                    for _ in range(len(self.shape) - len(other.shape)):
                        grad_to_add = grad_to_add.sum(axis=0)
                    # Handle shape differences
                    for i, (dim_self, dim_other) in enumerate(zip(self.shape[-len(other.shape):], other.shape)):
                        if dim_self != dim_other and dim_other == 1:
                            grad_to_add = grad_to_add.sum(axis=i-len(other.shape), keepdims=True)
                    other.grad.data += grad_to_add
        
        out._backward = _backward
        return out
    
    def __radd__(self, other: Union[Tensor, Any]) -> Tensor:
        return self + other
    
    def __sub__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="-"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += out.grad.data
            if other.requires_grad:
                if other.grad is not None:
                    # Handle broadcasting: sum over axes that were broadcasted
                    grad_to_add = out.grad.data
                    # If self has more dimensions than other, sum over the extra axes
                    for _ in range(len(self.shape) - len(other.shape)):
                        grad_to_add = grad_to_add.sum(axis=0)
                    # Handle shape differences
                    for i, (dim_self, dim_other) in enumerate(zip(self.shape[-len(other.shape):], other.shape)):
                        if dim_self != dim_other and dim_other == 1:
                            grad_to_add = grad_to_add.sum(axis=i-len(other.shape), keepdims=True)
                    other.grad.data -= grad_to_add
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self
    
    def __mul__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += other.data * out.grad.data
            if other.requires_grad:
                if other.grad is not None:
                    # Handle broadcasting: sum over axes that were broadcasted
                    grad_to_add = self.data * out.grad.data
                    # If self has more dimensions than other, sum over the extra axes
                    for _ in range(len(self.shape) - len(other.shape)):
                        grad_to_add = grad_to_add.sum(axis=0)
                    # Handle shape differences
                    for i, (dim_self, dim_other) in enumerate(zip(self.shape[-len(other.shape):], other.shape)):
                        if dim_self != dim_other and dim_other == 1:
                            grad_to_add = grad_to_add.sum(axis=i-len(other.shape), keepdims=True)
                    other.grad.data += grad_to_add
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other: Union[Tensor, Any]) -> Tensor:
        return self * other
    
    def __truediv__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="/"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad += out.grad / other.data
            if other.requires_grad:
                if other.grad is not None:
                    other.grad -= (self.data * out.grad) / (other.data ** 2)
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other: Union[Tensor, Any]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    def __pow__(self, power: Union[int, float]) -> Tensor:
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad += (power * (self.data ** (power - 1))) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> Tensor:
        return self * -1
    
    # Matrix operations
    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication."""
        out = Tensor(
            get_backend().xp.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += get_backend().xp.matmul(out.grad.data, other.data.T)
            if other.requires_grad:
                if other.grad is not None:
                    other.grad.data += get_backend().xp.matmul(self.data.T, out.grad.data)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)
    
    # Activation functions
    def relu(self) -> Tensor:
        """ReLU activation."""
        out = Tensor(
            maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += (self.data > 0) * out.grad.data
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> Tensor:
        """Sigmoid activation."""
        from .backend import exp
        sig = 1 / (1 + exp(-self.data))
        
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += (sig * (1 - sig)) * out.grad.data
        
        out._backward = _backward
        return out
    
    def tanh(self) -> Tensor:
        """Tanh activation."""
        from .backend import exp
        t = (exp(2 * self.data) - 1) / (exp(2 * self.data) + 1)
        
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += (1 - t ** 2) * out.grad.data
        
        out._backward = _backward
        return out
    
    def exp(self) -> Tensor:
        """Exponential function."""
        from .backend import exp
        out_data = exp(self.data)
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += out_data * out.grad.data
        
        out._backward = _backward
        return out
    
    def log(self) -> Tensor:
        """Natural logarithm."""
        from .backend import log
        out_data = log(self.data)
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += (1 / self.data) * out.grad.data
        
        out._backward = _backward
        return out
    
    # Reduction operations
    def sum(self, axis: Union[int, tuple[int, ...], None] = None, keepdims: bool = False) -> Tensor:
        """Sum reduction."""
        out_data = _sum(self.data, axis=axis, keepdims=keepdims)
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    # Expand gradient to match original shape
                    if axis is None:
                        # Sum over all axes - broadcast scalar gradient
                        self.grad.data += get_backend().xp.ones_like(self.data) * out.grad.data
                    else:
                        # Sum over specific axis
                        grad_expanded = out.grad.data
                        if not keepdims:
                            grad_expanded = get_backend().xp.expand_dims(out.grad.data, axis=axis)
                        # Broadcast along the summed axis
                        self.grad.data += get_backend().xp.ones_like(self.data) * grad_expanded
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Union[int, tuple[int, ...], None] = None, keepdims: bool = False) -> Tensor:
        """Mean reduction."""
        n = self.data.size if axis is None else self.data.shape[axis] if isinstance(axis, int) else 1
        return self.sum(axis=axis, keepdims=keepdims) / n
    
    def maximum(self, other: Union[Tensor, Any]) -> Tensor:
        """Element-wise maximum."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            get_backend().xp.maximum(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="maximum"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    mask = (self.data >= other.data).astype(float)
                    self.grad.data += mask * out.grad.data
            if other.requires_grad:
                if other.grad is not None:
                    mask = (other.data > self.data).astype(float)
                    other.grad.data += mask * out.grad.data
        
        out._backward = _backward
        return out
    
    def minimum(self, other: Union[Tensor, Any]) -> Tensor:
        """Element-wise minimum."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            get_backend().xp.minimum(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="minimum"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    mask = (self.data <= other.data).astype(float)
                    self.grad.data += mask * out.grad.data
            if other.requires_grad:
                if other.grad is not None:
                    mask = (other.data < self.data).astype(float)
                    other.grad.data += mask * out.grad.data
        
        out._backward = _backward
        return out
    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        """Reshape tensor."""
        out = Tensor(
            reshape(self.data, shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    self.grad.data += reshape(out.grad.data, self.data.shape)
        
        out._backward = _backward
        return out
    
    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        """Transpose tensor."""
        out = Tensor(
            get_backend().transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose"
        )
        
        def _backward() -> None:
            if self.requires_grad:
                if self.grad is not None:
                    # Inverse transpose
                    inv_axes = None
                    if axes is not None:
                        inv_axes = tuple(axes.index(i) for i in range(len(axes)))
                    self.grad.data += get_backend().transpose(out.grad.data, inv_axes)
        
        out._backward = _backward
        return out
    
    def T(self) -> Tensor:
        """Transpose (2D shortcut)."""
        return self.transpose()
