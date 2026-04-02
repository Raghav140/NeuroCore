import numpy as np
from typing import Tuple


def set_random_seed(seed: int = 42) -> None:
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    """
    assert 0.0 < test_size < 1.0
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy for classification.

    Supports:
    - Binary: y_true, y_pred in {0,1}
    - Multi-class: y_true as int labels, y_pred as probabilities/logits.
    """
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        else:
            y_true = y_true.reshape(-1)
    else:
        y_pred_labels = (y_pred.flatten() >= 0.5).astype(int)
        y_true = y_true.flatten().astype(int)

    return float(np.mean(y_true == y_pred_labels))


def iterate_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
):
    """
    Generator that yields mini-batches.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]

