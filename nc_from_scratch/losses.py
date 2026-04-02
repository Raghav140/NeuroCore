import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute scalar loss value."""
        raise NotImplementedError

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions."""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error loss for regression."""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(y_pred.shape)
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = y_true.reshape(y_pred.shape)
        # Do NOT average over batch here; Dense layer already
        # divides by batch size when computing parameter gradients.
        return 2.0 * (y_pred - y_true)


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss for binary classification."""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip for numerical stability
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        y_true = y_true.reshape(y_pred_clipped.shape)
        loss = -(
            y_true * np.log(y_pred_clipped)
            + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
        return float(np.mean(loss))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        y_true = y_true.reshape(y_pred_clipped.shape)
        grad = (-(y_true / y_pred_clipped) + (1.0 - y_true) / (1.0 - y_pred_clipped))
        # Do NOT average over batch here; Dense layer already
        # divides by batch size when computing parameter gradients.
        return grad


class CategoricalCrossEntropy(Loss):
    """
    Categorical cross-entropy loss for multi-class classification.

    Expects probabilities (typically softmax outputs) as y_pred and
    one-hot encoded labels or integer class indices as y_true.
    """

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # y_true given as integer class indices
            y_true_int = y_true.astype(int).reshape(-1)
            probs = y_pred_clipped[np.arange(y_pred_clipped.shape[0]), y_true_int]
        else:
            # one-hot encoded
            probs = np.sum(y_true * y_pred_clipped, axis=1)

        return float(-np.mean(np.log(probs)))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Gradient of CCE w.r.t. softmax probabilities.
        For the common softmax + CCE combination, the gradient
        w.r.t. logits simplifies to (y_pred - y_true) / N.
        """
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # integer class labels
            num_classes = y_pred.shape[1]
            y_true_oh = np.zeros_like(y_pred)
            y_true_int = y_true.astype(int).reshape(-1)
            y_true_oh[np.arange(y_pred.shape[0]), y_true_int] = 1.0
            y_true = y_true_oh

        # For softmax + CCE, gradient w.r.t. logits is (y_pred - y_true).
        # Do NOT average over batch here; Dense layer already
        # divides by batch size when computing parameter gradients.
        grad = (y_pred - y_true)
        return grad

