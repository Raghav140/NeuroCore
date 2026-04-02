import unittest

import numpy as np

import nnfs
from nnfs import BCELoss, Dense, ReLU, Sequential, Sigmoid, Trainer, set_backend
from nnfs.core.trainer import TrainerConfig
from nnfs.dashboard_logic import build_binary_model, decision_boundary_grid
from nnfs.optim import SGD
from nnfs.utils import make_xor


class TestNNFSLibrary(unittest.TestCase):
    def test_package_import(self):
        self.assertTrue(hasattr(nnfs, "__version__"))
        self.assertTrue(hasattr(nnfs, "Sequential"))

    def test_module_and_sequential_parameters(self):
        model = Sequential(Dense(2, 8), ReLU(), Dense(8, 1), Sigmoid())
        params = list(model.parameters())
        self.assertEqual(len(params), 4)

    def test_xor_training_with_trainer(self):
        np.random.seed(42)
        X, y = make_xor(300)
        model = Sequential(Dense(2, 16, init="he"), ReLU(), Dense(16, 1), Sigmoid())
        trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.1))
        trainer.fit(X, y, TrainerConfig(epochs=800, batch_size=64))
        preds = model(X)
        acc = ((preds.reshape(-1) >= 0.5).astype(int) == y.reshape(-1)).mean()
        self.assertGreaterEqual(float(acc), 0.95)

    def test_cpu_backend_consistency(self):
        set_backend("numpy")
        np.random.seed(7)
        X, y = make_xor(64)
        model = Sequential(Dense(2, 8), ReLU(), Dense(8, 1), Sigmoid())
        trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.1))
        trainer.fit(X, y, TrainerConfig(epochs=5, batch_size=32))
        p1 = np.asarray(model(X))

        set_backend("numpy")
        p2 = np.asarray(model(X))
        np.testing.assert_allclose(p1, p2, atol=1e-8)

    def test_dashboard_logic(self):
        X, _ = make_xor(40)
        model = build_binary_model(input_dim=2, hidden=8, activation="ReLU")
        xx, yy, zz = decision_boundary_grid(model, X, steps=30)
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(zz.shape, xx.shape)


if __name__ == "__main__":
    unittest.main()
