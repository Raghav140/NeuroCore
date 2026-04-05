"""Evaluation metrics."""

from __future__ import annotations

from typing import Tuple

from ..core.tensor import Tensor
from ..core.backend import sum, get_backend, abs, mean


def accuracy_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate accuracy score."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    correct = sum((y_true.data == y_pred.data).astype(int))
    total = y_true.shape[0]
    
    return float(correct / total)


def binary_accuracy(y_true: Tensor, y_pred_prob: Tensor, threshold: float = 0.5) -> float:
    """Calculate binary accuracy from probabilities."""
    y_pred = (y_pred_prob.data > threshold).astype(int)
    y_pred_tensor = Tensor(y_pred)
    
    return accuracy_score(y_true, y_pred_tensor)


def precision_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate precision score."""
    # True Positive / (True Positive + False Positive)
    y_true_np = y_true.data
    y_pred_np = y_pred.data
    
    true_positive = sum((y_true_np == 1) & (y_pred_np == 1))
    false_positive = sum((y_true_np == 0) & (y_pred_np == 1))
    
    if (true_positive + false_positive) == 0:
        return 0.0
    
    return float(true_positive / (true_positive + false_positive))


def recall_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate recall score."""
    # True Positive / (True Positive + False Negative)
    y_true_np = y_true.data
    y_pred_np = y_pred.data
    
    true_positive = sum((y_true_np == 1) & (y_pred_np == 1))
    false_negative = sum((y_true_np == 1) & (y_pred_np == 0))
    
    if (true_positive + false_negative) == 0:
        return 0.0
    
    return float(true_positive / (true_positive + false_negative))


def f1_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate F1 score."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if (precision + recall) == 0:
        return 0.0
    
    return float(2 * precision * recall / (precision + recall))


def f1_binary(y_true: Tensor, y_pred_prob: Tensor, threshold: float = 0.5) -> float:
    """Calculate F1 score for binary classification from probabilities."""
    y_pred = (y_pred_prob.data > threshold).astype(int)
    y_pred_tensor = Tensor(y_pred)
    
    return f1_score(y_true, y_pred_tensor)


def confusion_matrix_binary(y_true: Tensor, y_pred: Tensor) -> Tuple[int, int, int, int]:
    """Calculate confusion matrix for binary classification."""
    y_true_np = y_true.data
    y_pred_np = y_pred.data
    
    true_positive = int(sum((y_true_np == 1) & (y_pred_np == 1)))
    false_positive = int(sum((y_true_np == 0) & (y_pred_np == 1)))
    true_negative = int(sum((y_true_np == 0) & (y_pred_np == 0)))
    false_negative = int(sum((y_true_np == 1) & (y_pred_np == 0)))
    
    return true_positive, false_positive, true_negative, false_negative


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate mean squared error."""
    diff = y_true - y_pred
    mse = (diff * diff).mean()
    return float(mse.item())


def mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate mean absolute error."""
    diff = y_true - y_pred
    mae = abs(diff.data).mean()
    return float(mae)


def r2_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Calculate R² score (coefficient of determination)."""
    y_true_np = y_true.data
    y_pred_np = y_pred.data
    
    ss_res = sum((y_true_np - y_pred_np) ** 2)
    ss_tot = sum((y_true_np - mean(y_true_np)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return float(1 - (ss_res / ss_tot))


def classification_report(y_true: Tensor, y_pred: Tensor) -> str:
    """Generate a text report showing the main classification metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    tp, fp, tn, fn = confusion_matrix_binary(y_true, y_pred)
    
    report = f"""
Classification Report:
----------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}

Confusion Matrix:
------------------
              Predicted
              0    1
Actual 0   {tn:4d} {fp:4d}
       1   {fn:4d} {tp:4d}

True Positive:  {tp}
False Positive: {fp}
True Negative:  {tn}
False Negative: {fn}
"""
    return report
