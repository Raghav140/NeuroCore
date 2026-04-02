"""CLI entrypoint for nnfs."""

from __future__ import annotations

import argparse
import numpy as np

from .core import BCELoss, Sequential, Trainer
from .core.trainer import TrainerConfig
from .layers import Dense, ReLU, Sigmoid
from .optim import SGD
from .utils import accuracy_score, make_xor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple nnfs model.")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, y = make_xor(n_samples=400)

    model = Sequential(Dense(2, 16, init="he"), ReLU(), Dense(16, 1), Sigmoid())
    loss_fn = BCELoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    trainer = Trainer(model, loss_fn, optimizer)
    config = TrainerConfig(epochs=args.epochs, batch_size=args.batch_size)
    trainer.fit(X, y, config)

    preds = model(X)
    print(f"Final XOR accuracy: {accuracy_score(y, preds):.4f}")


if __name__ == "__main__":
    main()
