"""Core NNFS components."""

from .module import Module, Sequential
from .parameter import Parameter
from .tensor import Tensor
from .trainer import Trainer, TrainerConfig, TrainingHistory
from .backend import (
    set_backend, get_backend, get_backend_name, auto_detect_backend,
    array, zeros, ones, randn, sqrt, exp, log, maximum, sum, mean,
    reshape, transpose, concatenate, clip
)

__all__ = [
    "Module", "Sequential", "Parameter", "Tensor",
    "Trainer", "TrainerConfig", "TrainingHistory",
    "set_backend", "get_backend", "get_backend_name", "auto_detect_backend",
    "array", "zeros", "ones", "randn", "sqrt", "exp", "log", "maximum", "sum", "mean",
    "reshape", "transpose", "concatenate", "clip",
]
