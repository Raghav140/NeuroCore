"""Pure logic helpers used by Streamlit dashboard and tests."""

from __future__ import annotations

import numpy as np

from . import Dense, ReLU, Sequential, Sigmoid


def build_binary_model(input_dim: int, hidden: int, activation: str = "ReLU"):
    act = ReLU() if activation == "ReLU" else Sigmoid()
    return Sequential(Dense(input_dim, hidden, init="he"), act, Dense(hidden, 1), Sigmoid())


def decision_boundary_grid(model, X, steps: int = 120):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    out = model(grid)
    zz = (out.numpy() if hasattr(out, "numpy") else np.asarray(out)).reshape(xx.shape)
    return xx, yy, zz
