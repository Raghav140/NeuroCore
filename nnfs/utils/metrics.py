"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "data") and not isinstance(x, np.ndarray):
        try:
            from nnfs.core.backend import to_numpy

            return to_numpy(x.data)
        except Exception:
            pass
    return np.asarray(x)


def accuracy_score(y_true, y_pred) -> float:
    yp = _to_numpy(y_pred)
    yt = _to_numpy(y_true)
    if yp.ndim > 1 and yp.shape[1] > 1:
        pred = yp.argmax(axis=1)
        truth = yt.argmax(axis=1) if yt.ndim > 1 else yt.reshape(-1)
        return float((pred == truth).mean())
    pred = (yp.reshape(-1) >= 0.5).astype(int)
    truth = yt.reshape(-1).astype(int)
    return float((pred == truth).mean())


def confusion_matrix_binary(y_true, y_pred):
    pred = (_to_numpy(y_pred).reshape(-1) >= 0.5).astype(int)
    truth = _to_numpy(y_true).reshape(-1).astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


def f1_binary(y_true, y_pred) -> float:
    cm = confusion_matrix_binary(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
