"""Model zoo: simple CNN forward/backward sanity example."""

import numpy as np

from nnfs import BCELoss, Conv2D, Dense, Flatten, MaxPooling2D, ReLU, Sequential, Sigmoid, Trainer
from nnfs.core.trainer import TrainerConfig
from nnfs.optim import SGD

np.random.seed(42)
X = np.random.randn(128, 1, 8, 8).astype(np.float32)
y = (np.random.rand(128, 1) > 0.5).astype(np.float32)

model = Sequential(
    Conv2D(1, 4, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPooling2D(kernel_size=2),
    Flatten(),
    Dense(4 * 4 * 4, 16),
    ReLU(),
    Dense(16, 1),
    Sigmoid(),
)
trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.01))
history = trainer.fit(X, y, TrainerConfig(epochs=5, batch_size=32, debug_mode=True))
print("Expected: script runs, CNN autograd path is valid.")
print(f"Final loss: {history['loss'][-1]:.4f}")
