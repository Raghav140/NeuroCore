import numpy as np


def make_xor(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D XOR dataset.
    """
    X = np.random.randint(0, 2, size=(n_samples, 2)).astype(float)
    y = (X[:, 0] != X[:, 1]).astype(int).reshape(-1, 1)
    return X, y


def make_binary_classification(
    n_samples: int = 500, noise: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 2D binary classification dataset with two Gaussian blobs.
    """
    n_per_class = n_samples // 2
    mean0 = np.array([-1.0, -1.0])
    mean1 = np.array([1.0, 1.0])
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    X0 = np.random.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = np.random.multivariate_normal(mean1, cov, size=n_per_class)

    X = np.vstack([X0, X1])
    X += noise * np.random.randn(*X.shape)

    y = np.vstack(
        [np.zeros((n_per_class, 1), dtype=int), np.ones((n_per_class, 1), dtype=int)]
    )

    return X, y


def make_regression(
    n_samples: int = 500, noise: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D regression dataset: y = sin(x) + noise.
    """
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(*X.shape)
    return X, y

