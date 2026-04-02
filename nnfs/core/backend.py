"""Backend selection utilities (NumPy/CuPy)."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None

_backend = "numpy"


def is_cupy_available() -> bool:
    """Return True when CuPy is importable."""
    return cp is not None


def set_backend(name: str) -> None:
    """Set global backend: 'numpy', 'cupy', or 'auto'."""
    global _backend
    name = name.lower()
    if name == "auto":
        _backend = "cupy" if is_cupy_available() else "numpy"
        return
    if name not in {"numpy", "cupy"}:
        raise ValueError("Backend must be one of: numpy, cupy, auto")
    if name == "cupy" and not is_cupy_available():
        raise RuntimeError("CuPy backend requested but CuPy is not available")
    _backend = name


def get_backend_name() -> str:
    """Get the currently active backend name."""
    return _backend


def xp():
    """Return the active array module (numpy or cupy)."""
    if _backend == "cupy":
        return cp
    return np


def asarray(x):
    """Convert input to backend array."""
    return xp().asarray(x)


def to_numpy(x):
    """Convert array/tensor to NumPy ndarray."""
    if hasattr(x, "data"):
        # Works for nnfs Tensor/Parameter and keeps ndarray path intact.
        x = x.data
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)
