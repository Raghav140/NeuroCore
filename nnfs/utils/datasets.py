"""Synthetic datasets for demos and tests."""

from __future__ import annotations

import numpy as np


def make_xor(n_samples: int = 200):
    X = np.random.randint(0, 2, size=(n_samples, 2)).astype(float)
    y = (X[:, 0] != X[:, 1]).astype(int).reshape(-1, 1)
    return X, y


def make_binary_classification(n_samples: int = 500, noise: float = 0.2):
    n = n_samples // 2
    mean0, mean1 = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    X0 = np.random.multivariate_normal(mean0, cov, size=n)
    X1 = np.random.multivariate_normal(mean1, cov, size=n)
    X = np.vstack([X0, X1]) + noise * np.random.randn(2 * n, 2)
    y = np.vstack([np.zeros((n, 1), dtype=int), np.ones((n, 1), dtype=int)])
    return X, y


def make_regression(n_samples: int = 500, noise: float = 0.1):
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(*X.shape)
    return X, y
