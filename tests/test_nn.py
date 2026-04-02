import os
import tempfile
import unittest

import numpy as np

from nn_from_scratch.activations import ReLU, Sigmoid
from nn_from_scratch.dataset import make_xor
from nn_from_scratch.layers import Dense
from nn_from_scratch.losses import BinaryCrossEntropy
from nn_from_scratch.model import NeuralNetwork
from nn_from_scratch.optimizer import GradientDescent, ReduceLROnPlateau, StepLR
from nn_from_scratch.utils import accuracy, set_random_seed


class TestNeuralNetworkFramework(unittest.TestCase):
    def test_xor_training_reaches_high_accuracy(self):
        set_random_seed(42)
        X, y = make_xor(n_samples=400)

        model = NeuralNetwork()
        model.add(Dense(2, 8, weight_init="he"))
        model.add(ReLU())
        model.add(Dense(8, 1, weight_init="xavier"))
        model.add(Sigmoid())
        model.compile(loss=BinaryCrossEntropy(), optimizer=GradientDescent(lr=0.1))
        model.train(X, y, epochs=1200, batch_size=len(X), verbose=False)

        preds = model.predict(X)
        self.assertGreaterEqual(accuracy(y, preds), 0.95)

    def test_model_save_load_preserves_predictions(self):
        set_random_seed(42)
        X, y = make_xor(n_samples=200)

        model = NeuralNetwork()
        model.add(Dense(2, 8, weight_init="he"))
        model.add(ReLU())
        model.add(Dense(8, 1, weight_init="xavier"))
        model.add(Sigmoid())
        model.compile(loss=BinaryCrossEntropy(), optimizer=GradientDescent(lr=0.1))
        model.train(X, y, epochs=600, batch_size=len(X), verbose=False)
        original_preds = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "xor_model")
            model.save(prefix)
            loaded_model = NeuralNetwork.load(prefix)
            loaded_preds = loaded_model.predict(X)

        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-8, atol=1e-8)

    def test_step_lr_decreases_learning_rate(self):
        opt = GradientDescent(lr=0.1)
        scheduler = StepLR(opt, step_size=5, gamma=0.5)

        for epoch in range(1, 11):
            scheduler.step_epoch_start(epoch)

        self.assertAlmostEqual(opt.lr, 0.025, places=10)

    def test_reduce_on_plateau_decreases_learning_rate(self):
        opt = GradientDescent(lr=0.1)
        scheduler = ReduceLROnPlateau(
            opt, factor=0.5, patience=2, min_lr=1e-4, min_delta=1e-6
        )

        # First value sets baseline, then no improvement for 2 epochs -> decay.
        scheduler.step_metric(1.0)
        scheduler.step_metric(1.0)
        scheduler.step_metric(1.0)

        self.assertAlmostEqual(opt.lr, 0.05, places=10)


if __name__ == "__main__":
    unittest.main()
