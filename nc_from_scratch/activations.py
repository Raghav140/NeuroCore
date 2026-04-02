import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation."""

    def __init__(self):
        self._inputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self._inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output.copy()
        grad_input[self._inputs <= 0] = 0.0
        return grad_input


class Sigmoid(Activation):
    """Sigmoid activation."""

    def __init__(self):
        self._outputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Numerically stable sigmoid
        positive_mask = inputs >= 0
        negative_mask = ~positive_mask

        z = np.zeros_like(inputs)
        z[positive_mask] = np.exp(-inputs[positive_mask])
        z[negative_mask] = np.exp(inputs[negative_mask])

        outputs = np.zeros_like(inputs)
        outputs[positive_mask] = 1.0 / (1.0 + z[positive_mask])
        outputs[negative_mask] = z[negative_mask] / (1.0 + z[negative_mask])

        self._outputs = outputs
        return outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._outputs * (1.0 - self._outputs)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def __init__(self):
        self._outputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        outputs = np.tanh(inputs)
        self._outputs = outputs
        return outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - self._outputs ** 2)


class Softmax(Activation):
    """
    Softmax activation.

    This implementation is numerically stable.
    Note: In practice, for classification, softmax is often combined
    with categorical cross-entropy loss. Here we provide a generic
    activation with forward and backward for completeness.
    """

    def __init__(self):
        self._outputs = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Shift inputs by max for numerical stability
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        sums = np.sum(exp_vals, axis=1, keepdims=True)
        outputs = exp_vals / sums
        self._outputs = outputs
        return outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax.

        Computes the gradient of the loss w.r.t. the inputs given
        the gradient w.r.t. the outputs.
        """
        batch_size, num_classes = self._outputs.shape
        grad_input = np.empty_like(grad_output)

        # Compute Jacobian-vector product for each sample in the batch
        for i in range(batch_size):
            s = self._outputs[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            grad_input[i] = jacobian @ grad_output[i]

        return grad_input

