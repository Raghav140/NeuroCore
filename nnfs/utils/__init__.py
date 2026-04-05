"""Utility functions."""

from .datasets import (
    make_binary_classification, make_xor, make_moons, make_spiral,
    train_test_split, standardize, normalize
)
from .metrics import (
    accuracy_score, binary_accuracy, precision_score, recall_score,
    f1_score, f1_binary, confusion_matrix_binary,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)
from .benchmark import Benchmark, benchmark_model, compare_models
from .summary import model_summary, print_model_summary

__all__ = [
    "make_binary_classification", "make_xor", "make_moons", "make_spiral",
    "train_test_split", "standardize", "normalize",
    "accuracy_score", "binary_accuracy", "precision_score", "recall_score",
    "f1_score", "f1_binary", "confusion_matrix_binary",
    "mean_squared_error", "mean_absolute_error", "r2_score",
    "classification_report",
    "Benchmark", "benchmark_model", "compare_models",
    "model_summary", "print_model_summary",
]
