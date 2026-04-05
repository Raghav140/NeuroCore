import unittest

import numpy as np

import nnfs
from nnfs import BCELoss, Dense, ReLU, Sequential, Sigmoid, Trainer, set_backend
from nnfs.core.trainer import TrainerConfig
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
        model = Sequential(Dense(2, 16), ReLU(), Dense(16, 1), Sigmoid())
        config = TrainerConfig(epochs=800, batch_size=64, learning_rate=0.1)
        trainer = Trainer(model, BCELoss(), SGD(list(model.parameters()), lr=0.1), config)
        trainer.fit(X, y)
        preds = model(X)
        acc = ((preds.data.reshape(-1) >= 0.5).astype(int) == y.data.reshape(-1)).mean()
        self.assertGreaterEqual(float(acc), 0.95)

    def test_cpu_backend_consistency(self):
        set_backend("numpy")
        np.random.seed(7)
        X, y = make_xor(64)
        model = Sequential(Dense(2, 8), ReLU(), Dense(8, 1), Sigmoid())
        trainer = Trainer(model, BCELoss(), SGD(list(model.parameters()), lr=0.1))
        trainer.fit(X, y, TrainerConfig(epochs=5, batch_size=32))
        p1 = model(X).data

        set_backend("numpy")
        p2 = model(X).data
        np.testing.assert_allclose(p1, p2, atol=1e-8)

    def test_dashboard_logic(self):
        """Test dashboard helper functions."""
        from nnfs.utils.dashboard import build_binary_model, get_default_training_config, prepare_binary_dataset
        
        # Test model building
        model = build_binary_model(input_size=2, hidden_size=32)
        self.assertIsNotNone(model)
        
        # Test default config
        config = get_default_training_config()
        self.assertIn('epochs', config)
        self.assertIn('batch_size', config)
        self.assertIn('learning_rate', config)
        
        # Test dataset preparation
        X, y = prepare_binary_dataset(n_samples=100)
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(y.shape[0], 100)
        
        # Test model summary
        from nnfs.utils.dashboard import get_model_summary
        summary = get_model_summary(model)
        self.assertIn('total_parameters', summary)
        self.assertGreater(summary['total_parameters'], 0)


if __name__ == "__main__":
    unittest.main()
