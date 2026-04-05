"""Backend abstraction for NumPy/CuPy operations."""

from __future__ import annotations

from typing import Any, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

import numpy as np


class Backend:
    """Backend abstraction for array operations."""
    
    def __init__(self, name: str = "numpy"):
        self.name = name
        if name == "cupy" and not CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        self.xp = cp if name == "cupy" else np
    
    def as_numpy(self, array: Any) -> np.ndarray:
        """Convert array to NumPy regardless of backend."""
        if self.name == "cupy" and hasattr(array, 'get'):
            return array.get()
        return array
    
    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert NumPy array to backend array."""
        if self.name == "cupy":
            return cp.asarray(array)
        return array


# Global backend instance
_backend: Backend = Backend("numpy")


def set_backend(name: str) -> None:
    """Set the global backend."""
    global _backend
    _backend = Backend(name)


def get_backend_name() -> str:
    """Get the current backend name."""
    return _backend.name


def get_backend() -> Backend:
    """Get the current backend instance."""
    return _backend


def auto_detect_backend() -> str:
    """Auto-detect the best available backend."""
    if CUPY_AVAILABLE:
        return "cupy"
    return "numpy"


# Convenience functions that delegate to backend
def array(data: Any, dtype: Any = None) -> Any:
    """Create an array using the current backend."""
    return _backend.xp.array(data, dtype=dtype)


def zeros(shape: tuple[int, ...], dtype: Any = None) -> Any:
    """Create zeros array using the current backend."""
    return _backend.xp.zeros(shape, dtype=dtype)


def ones(shape: tuple[int, ...], dtype: Any = None) -> Any:
    """Create ones array using the current backend."""
    return _backend.xp.ones(shape, dtype=dtype)


def randn(*shape: int, dtype: Any = None) -> Any:
    """Create random normal array using the current backend."""
    if _backend.name == "cupy":
        return _backend.xp.random.randn(*shape, dtype=dtype)
    return _backend.xp.random.randn(*shape)


def sqrt(x: Any) -> Any:
    """Square root using current backend."""
    return _backend.xp.sqrt(x)


def exp(x: Any) -> Any:
    """Exponential using current backend."""
    return _backend.xp.exp(x)


def log(x: Any) -> Any:
    """Natural logarithm using current backend."""
    return _backend.xp.log(x)


def maximum(x1: Any, x2: Any) -> Any:
    """Element-wise maximum using current backend."""
    return _backend.xp.maximum(x1, x2)


def sum(x: Any, axis: Union[int, tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
    """Sum using current backend."""
    return _backend.xp.sum(x, axis=axis, keepdims=keepdims)


def mean(x: Any, axis: Union[int, tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
    """Mean using current backend."""
    return _backend.xp.mean(x, axis=axis, keepdims=keepdims)


def reshape(x: Any, shape: tuple[int, ...]) -> Any:
    """Reshape array using current backend."""
    return _backend.xp.reshape(x, shape)


def transpose(x: Any, axes: tuple[int, ...] | None = None) -> Any:
    """Transpose array using current backend."""
    return _backend.xp.transpose(x, axes)


def concatenate(arrays: list[Any], axis: int = 0) -> Any:
    """Concatenate arrays using current backend."""
    return _backend.xp.concatenate(arrays, axis=axis)


def zeros_like(x: Any, dtype: Any = None) -> Any:
    """Create zeros array like input using current backend."""
    return _backend.xp.zeros_like(x, dtype=dtype)


def ones_like(x: Any, dtype: Any = None) -> Any:
    """Create ones array like input using current backend."""
    return _backend.xp.ones_like(x, dtype=dtype)


def clip(x: Any, min_val: Any, max_val: Any) -> Any:
    """Clip array values using current backend."""
    return _backend.xp.clip(x, min_val, max_val)


def maximum(x1: Any, x2: Any) -> Any:
    """Element-wise maximum using current backend."""
    return _backend.xp.maximum(x1, x2)


def minimum(x1: Any, x2: Any) -> Any:
    """Element-wise minimum using current backend."""
    return _backend.xp.minimum(x1, x2)


def abs(x: Any) -> Any:
    """Element-wise absolute value using current backend."""
    return _backend.xp.abs(x)


def mean(x: Any, axis: Any = None) -> Any:
    """Compute mean using current backend."""
    return _backend.xp.mean(x, axis=axis)
