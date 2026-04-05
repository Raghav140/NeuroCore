"""NNFS - Neural Network From Scratch

A clean, educational deep learning framework implemented with NumPy (and optional CuPy GPU support),
featuring a PyTorch-inspired API.
"""

from .core.module import Module, Sequential
from .core.parameter import Parameter
from .core.tensor import Tensor
from .core.trainer import Trainer, TrainerConfig, TrainingHistory
from .core.backend import set_backend, get_backend_name, auto_detect_backend

# Layers
from .layers.dense import Dense
from .layers.activations import ReLU, Sigmoid, Tanh, Softmax
from .layers.normalization import BatchNorm1d
from .layers.dropout import Dropout
from .layers.conv import Conv2D, Flatten
from .layers.pooling import MaxPooling2D, AvgPooling2D
from .layers.losses import MSELoss, BCELoss, CrossEntropyLoss

# Optimizers
from .optim.sgd import SGD, Adam

# Schedulers
from .optim.scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

# Utilities
from .utils.datasets import (
    make_binary_classification, make_xor, make_moons, make_spiral,
    train_test_split, standardize, normalize
)
from .utils.metrics import (
    accuracy_score, binary_accuracy, precision_score, recall_score,
    f1_score, f1_binary, confusion_matrix_binary,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)
from .utils.benchmark import Benchmark, benchmark_model, compare_models
from .utils.summary import model_summary, print_model_summary

__version__ = "0.1.0"
__author__ = "Raghav Sharma"

__all__ = [
    # Core
    "Module", "Sequential", "Parameter", "Tensor",
    "Trainer", "TrainerConfig", "TrainingHistory",
    "set_backend", "get_backend_name", "auto_detect_backend",
    
    # Layers
    "Dense", "ReLU", "Sigmoid", "Tanh", "Softmax",
    "BatchNorm1d", "Dropout", "Conv2D", "Flatten",
    "MaxPooling2D", "AvgPooling2D",
    
    # Losses
    "MSELoss", "BCELoss", "CrossEntropyLoss",
    
    # Optimizers
    "SGD", "Adam",
    
    # Schedulers
    "StepLR", "ExponentialLR", "ReduceLROnPlateau",
    
    # Utilities
    "make_binary_classification", "make_xor", "make_moons", "make_spiral",
    "train_test_split", "standardize", "normalize",
    "accuracy_score", "binary_accuracy", "precision_score", "recall_score",
    "f1_score", "f1_binary", "confusion_matrix_binary",
    "mean_squared_error", "mean_absolute_error", "r2_score",
    "classification_report",
    "Benchmark", "benchmark_model", "compare_models",
    "model_summary", "print_model_summary",
]
