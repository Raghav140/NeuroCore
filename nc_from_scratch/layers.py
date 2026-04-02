import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for all layers."""

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the forward pass."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute the backward pass (gradient wrt inputs)."""
        raise NotImplementedError

    @property
    def params(self):
        """Return learnable parameters as a dict."""
        return {}

    @property
    def grads(self):
        """Return gradients of learnable parameters as a dict."""
        return {}


class Dense(Layer):
    """
    Fully connected (dense) layer.

    Supports Xavier and He initialization depending on activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_init: str = "xavier",
        l2_lambda: float = 0.0,
    ):
        """
        Initialize a Dense layer.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output features.
        weight_init : {"xavier", "he"}
            Weight initialization strategy.
        l2_lambda : float
            L2 regularization strength.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_lambda = l2_lambda

        if weight_init == "xavier":
            limit = np.sqrt(2.0 / (input_dim + output_dim))
        elif weight_init == "he":
            limit = np.sqrt(2.0 / input_dim)
        else:
            raise ValueError("weight_init must be 'xavier' or 'he'")

        self.W = np.random.randn(input_dim, output_dim) * limit
        self.b = np.zeros((1, output_dim))

        self._inputs = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        inputs : np.ndarray
            Input tensor of shape (batch_size, input_dim).
        training : bool
            Whether in training mode (unused, for API consistency).
        """
        self._inputs = inputs
        return inputs @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of loss w.r.t. outputs, shape (batch_size, output_dim).
        """
        batch_size = self._inputs.shape[0]

        self.dW = self._inputs.T @ grad_output / batch_size
        self.db = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        if self.l2_lambda > 0.0:
            self.dW += self.l2_lambda * self.W

        grad_input = grad_output @ self.W.T
        return grad_input

    @property
    def params(self):
        return {"W": self.W, "b": self.b}

    @property
    def grads(self):
        return {"W": self.dW, "b": self.db}


class Dropout(Layer):
    """
    Dropout layer for regularization.
    """

    def __init__(self, rate: float):
        """
        Parameters
        ----------
        rate : float
            Dropout rate in [0, 1). Fraction of units to drop.
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self._mask = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        self._mask = (np.random.rand(*inputs.shape) < keep_prob) / keep_prob
        return inputs * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.rate == 0.0 or self._mask is None:
            return grad_output
        return grad_output * self._mask

