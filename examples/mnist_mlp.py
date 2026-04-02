"""Model zoo: MNIST-like MLP scaffold (uses random data fallback)."""

import numpy as np

from nnfs import CrossEntropyLoss, Dense, ReLU, Sequential, Softmax, Trainer
from nnfs.core.trainer import TrainerConfig
from nnfs.optim import SGD


def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


np.random.seed(42)
# Lightweight placeholder dataset for portability (replace with true MNIST loader).
X = np.random.randn(2000, 28 * 28).astype(np.float32)
y_idx = np.random.randint(0, 10, size=(2000,))
y = one_hot(y_idx, 10)

model = Sequential(Dense(28 * 28, 128, init="he"), ReLU(), Dense(128, 10), Softmax())
trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.05))
history = trainer.fit(X, y, TrainerConfig(epochs=20, batch_size=128))
print("Expected: script runs and loss decreases over epochs.")
print(f"Final training loss: {history['loss'][-1]:.4f}")
