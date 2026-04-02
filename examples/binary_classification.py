"""Model zoo: synthetic binary classification example."""

import numpy as np

from nnfs import BCELoss, Dense, ReLU, Sequential, Sigmoid, Trainer
from nnfs.core.trainer import TrainerConfig
from nnfs.optim import SGD, StepLR
from nnfs.utils import accuracy_score, make_binary_classification

np.random.seed(42)
X, y = make_binary_classification(800, noise=0.2)
model = Sequential(Dense(2, 32, init="he"), ReLU(), Dense(32, 1), Sigmoid())
opt = SGD(model.parameters(), lr=0.05, momentum=0.9)
sched = StepLR(opt, step_size=50, gamma=0.8)
trainer = Trainer(model, BCELoss(), opt, scheduler=sched)
trainer.fit(X, y, TrainerConfig(epochs=200, batch_size=64))
preds = model(X).numpy()
print("Expected: accuracy around 0.85+")
print(f"Observed accuracy: {accuracy_score(y, preds):.4f}")
