"""Dashboard utilities for NNFS."""

from __future__ import annotations

from typing import Dict, Any, Optional

import nnfs
from nnfs.layers import Dense, ReLU, Sigmoid, Dropout
from nnfs.optim import SGD
from nnfs.utils import make_binary_classification


def build_binary_model(input_size: int = 2, hidden_size: int = 64, dropout_rate: float = 0.0):
    """Build a binary classification model for dashboard/testing purposes.
    
    Args:
        input_size: Number of input features
        hidden_size: Size of hidden layers
        dropout_rate: Dropout rate (0.0 to disable dropout)
    
    Returns:
        A Sequential model ready for binary classification
    """
    layers = [
        Dense(input_size, hidden_size),
        ReLU(),
    ]
    
    if dropout_rate > 0.0:
        layers.append(Dropout(dropout_rate))
    
    layers.extend([
        Dense(hidden_size, hidden_size // 2),
        ReLU(),
    ])
    
    if dropout_rate > 0.0:
        layers.append(Dropout(dropout_rate))
    
    layers.extend([
        Dense(hidden_size // 2, 1),
        Sigmoid()
    ])
    
    return nnfs.Sequential(*layers)


def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration for dashboard."""
    return {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.01,
        'momentum': 0.0,
        'weight_decay': 0.0,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'log_interval': 5
    }


def prepare_binary_dataset(n_samples: int = 1000, n_features: int = 2, 
                          noise: float = 0.1, random_state: int = 42):
    """Prepare a binary classification dataset for dashboard.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        noise: Amount of noise to add
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    return make_binary_classification(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )


def create_trainer_components(model, learning_rate: float = 0.01, momentum: float = 0.0):
    """Create optimizer and loss function for training.
    
    Args:
        model: The model to create components for
        learning_rate: Learning rate for optimizer
        momentum: Momentum for SGD optimizer
    
    Returns:
        Tuple of (optimizer, loss_fn)
    """
    optimizer = SGD(
        list(model.parameters()),
        lr=learning_rate,
        momentum=momentum
    )
    loss_fn = nnfs.BCELoss()
    
    return optimizer, loss_fn


def get_model_summary(model) -> Dict[str, Any]:
    """Get a summary of the model architecture.
    
    Args:
        model: The model to summarize
    
    Returns:
        Dictionary containing model information
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.data.size
        if param.requires_grad:
            trainable_params += param.data.size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': 'Sequential',
        'num_layers': len(list(model.children()))
    }
