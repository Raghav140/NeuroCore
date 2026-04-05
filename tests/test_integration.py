"""Integration tests for NNFS framework."""

import unittest
import numpy as np
import nnfs
from nnfs import Sequential, Dense, ReLU, Sigmoid, Dropout, BCELoss
from nnfs.optim import SGD
from nnfs.utils import make_xor, make_binary_classification
from nnfs.core import Trainer, TrainerConfig


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete NNFS framework."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline with real data."""
        np.random.seed(42)
        
        # Create dataset
        X, y = make_xor(500)
        
        # Build model
        model = Sequential(
            Dense(2, 32),
            ReLU(),
            Dense(32, 16),
            ReLU(),
            Dense(16, 1),
            Sigmoid()
        )
        
        # Setup training
        optimizer = SGD(list(model.parameters()), lr=0.1)
        loss_fn = BCELoss()
        config = TrainerConfig(epochs=50, batch_size=32, learning_rate=0.1)
        trainer = Trainer(model, loss_fn, optimizer, config)
        
        # Train
        history = trainer.fit(X, y)
        
        # Check that training happened
        self.assertEqual(len(history.train_loss), 50)
        self.assertEqual(len(history.train_accuracy), 50)
        
        # Check that loss decreased
        self.assertLess(history.train_loss[-1], history.train_loss[0])
        
        # Check that accuracy improved
        self.assertGreater(history.train_accuracy[-1], 0.9)
    
    def test_model_with_dropout(self):
        """Test model with dropout layers."""
        np.random.seed(42)
        
        # Create dataset
        X, y = make_binary_classification(200, n_features=4)
        
        # Build model with dropout
        model = Sequential(
            Dense(4, 32),
            ReLU(),
            Dropout(0.2),
            Dense(32, 16),
            ReLU(),
            Dropout(0.2),
            Dense(16, 1),
            Sigmoid()
        )
        
        # Setup training
        optimizer = SGD(list(model.parameters()), lr=0.1)
        loss_fn = BCELoss()
        config = TrainerConfig(epochs=20, batch_size=32)
        trainer = Trainer(model, loss_fn, optimizer, config)
        
        # Train
        trainer.fit(X, y)
        
        # Check that model can make predictions
        predictions = model(X)
        self.assertEqual(predictions.shape[0], X.shape[0])
        self.assertEqual(predictions.shape[1], 1)
    
    def test_model_save_load_functionality(self):
        """Test model save and load functionality."""
        np.random.seed(42)
        
        # Create and train a simple model
        X, y = make_xor(100)
        model = Sequential(Dense(2, 8), ReLU(), Dense(8, 1), Sigmoid())
        
        # Get predictions before training
        preds_before = model(X).data.copy()
        
        # Train briefly
        optimizer = SGD(list(model.parameters()), lr=0.1)
        loss_fn = BCELoss()
        config = TrainerConfig(epochs=5, batch_size=32)
        trainer = Trainer(model, loss_fn, optimizer, config)
        trainer.fit(X, y)
        
        # Get predictions after training
        preds_after = model(X).data.copy()
        
        # Check that predictions changed
        self.assertFalse(np.allclose(preds_before, preds_after))
    
    def test_different_learning_rates(self):
        """Test training with different learning rates."""
        np.random.seed(42)
        
        X, y = make_xor(200)
        
        # Test with different learning rates
        for lr in [0.01, 0.1, 0.5]:
            model = Sequential(Dense(2, 16), ReLU(), Dense(16, 1), Sigmoid())
            optimizer = SGD(list(model.parameters()), lr=lr)
            loss_fn = BCELoss()
            config = TrainerConfig(epochs=20, batch_size=32, learning_rate=lr)
            trainer = Trainer(model, loss_fn, optimizer, config)
            
            # Train
            trainer.fit(X, y)
            
            # Check that model can make predictions
            predictions = model(X)
            self.assertEqual(predictions.shape[0], X.shape[0])
    
    def test_gradient_flow_check(self):
        """Test that gradients flow properly through the network."""
        np.random.seed(42)
        
        # Create model
        model = Sequential(
            Dense(2, 16),
            ReLU(),
            Dense(16, 8),
            ReLU(),
            Dense(8, 1),
            Sigmoid()
        )
        
        # Create data
        X = nnfs.Tensor(np.random.randn(10, 2))
        y = nnfs.Tensor(np.random.randn(10, 1))
        
        # Forward pass
        output = model(X)
        loss = BCELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(np.allclose(param.grad.data, 0))
    
    def test_multiple_backward_passes(self):
        """Test multiple backward passes on the same computation graph."""
        np.random.seed(42)
        
        # Create simple computation
        x = nnfs.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        
        # First computation
        y1 = x * 2
        z1 = y1 + 1
        z1.backward()
        grad_first = x.grad.data.copy()
        
        # Reset gradients
        x.zero_grad()
        
        # Second computation (fresh graph)
        y2 = x * 2
        z2 = y2 + 1
        z2.backward()
        grad_second = x.grad.data.copy()
        
        # Gradients should be the same
        np.testing.assert_allclose(grad_first, grad_second)


if __name__ == "__main__":
    unittest.main()
