import unittest

import numpy as np

from nnfs import BCELoss, Conv2D, Dense, Flatten, MaxPooling2D, ReLU, Sequential, Sigmoid, Tensor
from nnfs.optim import SGD
from nnfs.utils import benchmark_backends


class TestAutogradAndCNN(unittest.TestCase):
    def test_tensor_autograd_scalar(self):
        x = Tensor(np.array([[2.0]]), requires_grad=True)
        y = (x * x + x).sum()
        y.backward()
        self.assertAlmostEqual(float(x.grad.reshape(-1)[0]), 5.0, places=5)

    def test_cnn_forward_backward(self):
        np.random.seed(0)
        X = np.random.randn(16, 1, 8, 8).astype(np.float32)
        y = (np.random.rand(16, 1) > 0.5).astype(np.float32)
        model = Sequential(
            Conv2D(1, 2, kernel_size=3, padding=1),
            ReLU(),
            MaxPooling2D(2),
            Flatten(),
            Dense(2 * 4 * 4, 1),
            Sigmoid(),
        )
        opt = SGD(model.parameters(), lr=0.01)
        opt.zero_grad()
        preds = model(X)
        loss = BCELoss()(preds, y)
        loss.backward()
        # Gradients should be populated for trainable params
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_benchmark_runs(self):
        results = benchmark_backends(backends=["numpy"], epochs=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("forward_ms", results[0])


if __name__ == "__main__":
    unittest.main()
