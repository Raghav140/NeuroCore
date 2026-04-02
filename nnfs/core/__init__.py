"""Core exports."""

from .backend import asarray, get_backend_name, is_cupy_available, set_backend, to_numpy, xp
from .containers import Sequential
from .losses import BCELoss, CrossEntropyLoss, MSELoss
from .module import Module, Parameter
from .tensor import Tensor, tensor
from .trainer import Callback, Trainer
