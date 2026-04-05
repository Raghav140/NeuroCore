"""Dataset utilities and synthetic data generators."""

from __future__ import annotations

from typing import Tuple, Optional

from ..core.tensor import Tensor
from ..core.backend import get_backend, array


def make_binary_classification(
    n_samples: int = 1000,
    n_features: int = 2,
    n_classes: int = 2,
    random_state: Optional[int] = None,
    noise: float = 0.1
) -> Tuple[Tensor, Tensor]:
    """Generate synthetic binary classification dataset."""
    if random_state is not None:
        get_backend().xp.random.seed(random_state)
    
    # Generate random data points
    X = get_backend().xp.random.randn(n_samples, n_features)
    
    # Create labels based on a simple decision boundary
    if n_features == 2:
        # Circular decision boundary for visualization
        y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(int)
    else:
        # Linear decision boundary for higher dimensions
        weights = get_backend().xp.random.randn(n_features)
        y = ((X @ weights) > 0).astype(int)
    
    # Add noise
    if noise > 0:
        X += noise * get_backend().xp.random.randn(*X.shape)
    
    # Convert to tensors
    X_tensor = Tensor(X.astype(float))
    y_tensor = Tensor(y.astype(float))
    
    return X_tensor, y_tensor


def make_xor(n_samples: int = 200, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generate XOR dataset."""
    if random_state is not None:
        get_backend().xp.random.seed(random_state)
    
    # Generate XOR data
    choices = get_backend().xp.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    indices = get_backend().xp.random.choice(len(choices), size=n_samples)
    X = choices[indices]
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    
    # Add noise
    if noise > 0:
        X = X.astype(float) + noise * get_backend().xp.random.randn(*X.shape)
    
    # Convert to tensors
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.astype(float))
    
    return X_tensor, y_tensor


def make_moons(n_samples: int = 1000, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generate two moons dataset."""
    if random_state is not None:
        get_backend().xp.random.seed(random_state)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Generate outer circle
    outer_linspace = get_backend().xp.linspace(0, get_backend().xp.pi, n_samples_out)
    outer_x = get_backend().xp.cos(outer_linspace)
    outer_y = get_backend().xp.sin(outer_linspace)
    
    # Generate inner circle
    inner_linspace = get_backend().xp.linspace(0, get_backend().xp.pi, n_samples_in)
    inner_x = 1 - get_backend().xp.cos(inner_linspace)
    inner_y = 1 - get_backend().xp.sin(inner_linspace) - 0.5
    
    # Combine
    X = get_backend().xp.vstack([get_backend().xp.column_stack([outer_x, outer_y]),
                                 get_backend().xp.column_stack([inner_x, inner_y])])
    y = get_backend().xp.hstack([get_backend().xp.zeros(n_samples_out),
                                 get_backend().xp.ones(n_samples_in)])
    
    # Add noise
    if noise > 0:
        X += noise * get_backend().xp.random.randn(*X.shape)
    
    # Convert to tensors
    X_tensor = Tensor(X.astype(float))
    y_tensor = Tensor(y.astype(float))
    
    return X_tensor, y_tensor


def make_spiral(n_samples: int = 1000, noise: float = 0.1, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Generate spiral dataset."""
    if random_state is not None:
        get_backend().xp.random.seed(random_state)
    
    n_samples_per_class = n_samples // 2
    
    # Generate first spiral
    t1 = get_backend().xp.linspace(0, 4 * get_backend().xp.pi, n_samples_per_class)
    x1 = t1 * get_backend().xp.cos(t1) + noise * get_backend().xp.random.randn(n_samples_per_class)
    y1 = t1 * get_backend().xp.sin(t1) + noise * get_backend().xp.random.randn(n_samples_per_class)
    
    # Generate second spiral
    t2 = get_backend().xp.linspace(0, 4 * get_backend().xp.pi, n_samples_per_class)
    x2 = t2 * get_backend().xp.cos(t2) + noise * get_backend().xp.random.randn(n_samples_per_class)
    y2 = t2 * get_backend().xp.sin(t2) + noise * get_backend().xp.random.randn(n_samples_per_class)
    
    # Combine
    X = get_backend().xp.vstack([get_backend().xp.column_stack([x1, y1]),
                                 get_backend().xp.column_stack([x2, y2])])
    y = get_backend().xp.hstack([get_backend().xp.zeros(n_samples_per_class),
                                 get_backend().xp.ones(n_samples_per_class)])
    
    # Convert to tensors
    X_tensor = Tensor(X.astype(float))
    y_tensor = Tensor(y.astype(float))
    
    return X_tensor, y_tensor


def train_test_split(X: Tensor, y: Tensor, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split data into training and testing sets."""
    if random_state is not None:
        get_backend().xp.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = get_backend().xp.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = Tensor(X.data[train_indices])
    X_test = Tensor(X.data[test_indices])
    y_train = Tensor(y.data[train_indices])
    y_test = Tensor(y.data[test_indices])
    
    return X_train, X_test, y_train, y_test


def standardize(X: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Standardize features by removing mean and scaling to unit variance."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    
    # Avoid division by zero
    std_safe = get_backend().maximum(std.data, 1e-8)
    
    X_standardized = (X - mean) / Tensor(std_safe)
    
    return X_standardized, mean, std


def normalize(X: Tensor, min_val: Optional[Tensor] = None, max_val: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Normalize features to range [0, 1]."""
    if min_val is None:
        min_val = X.min(axis=0)
    if max_val is None:
        max_val = X.max(axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_safe = get_backend().maximum(range_val.data, 1e-8)
    
    X_normalized = (X - min_val) / Tensor(range_safe)
    
    return X_normalized, min_val, max_val
