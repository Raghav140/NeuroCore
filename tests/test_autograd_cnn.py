import unittest

import numpy as np

from nnfs import BCELoss, Conv2D, Dense, Flatten, MaxPooling2D, ReLU, Sequential, Sigmoid, Tensor
from nnfs.optim import SGD
# from nnfs.utils import benchmark_backends  # TODO: Implement this function


class TestAutogradAndCNN(unittest.TestCase):
    def test_tensor_autograd_scalar(self):
        x = Tensor(np.array([[2.0]]), requires_grad=True)
        y = (x * x + x).sum()
        y.backward()
        self.assertAlmostEqual(float(x.grad.data.reshape(-1)[0]), 5.0, places=5)

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
        """Test that benchmark function runs without errors."""
        from nnfs.utils.benchmark import benchmark_backends
        import nnfs
        from nnfs import Sequential, Dense, ReLU, Sigmoid
        from nnfs.optim import SGD
        from nnfs.utils import make_xor
        
        # Create simple model and data
        model = Sequential(Dense(2, 4), ReLU(), Dense(4, 1), Sigmoid())
        X, y = make_xor(50)
        optimizer = SGD(list(model.parameters()), lr=0.01)
        loss_fn = nnfs.BCELoss()
        
        # Run benchmark
        results = benchmark_backends(
            model, X, y, loss_fn, optimizer, 
            backends=['numpy'], 
            epochs=2
        )
        
        # Check results structure
        self.assertIn('numpy', results)
        self.assertIn('forward_time', results['numpy'])
        self.assertIn('backward_time', results['numpy'])
        self.assertIn('epoch_time', results['numpy'])
        
        # Check that times are positive
        self.assertGreater(results['numpy']['forward_time'], 0)
        self.assertGreater(results['numpy']['backward_time'], 0)
        self.assertGreater(results['numpy']['epoch_time'], 0)


if __name__ == "__main__":
    unittest.main()
