"""Model zoo: XOR classification example."""

import numpy as np

from nnfs import BCELoss, Dense, ReLU, Sequential, Sigmoid, Trainer
from nnfs.core.trainer import TrainerConfig
from nnfs.optim import SGD
from nnfs.utils import accuracy_score, make_xor

np.random.seed(42)
X, y = make_xor(400)
model = Sequential(Dense(2, 16, init="he"), ReLU(), Dense(16, 1), Sigmoid())
trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.1))
trainer.fit(X, y, TrainerConfig(epochs=700, batch_size=64))
preds = model(X).numpy()
print("Expected: XOR accuracy > 0.95")
print(f"Observed accuracy: {accuracy_score(y, preds):.4f}")
