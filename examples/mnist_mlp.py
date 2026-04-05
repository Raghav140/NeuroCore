"""MNIST MLP example using NNFS (with synthetic data)."""

import nnfs
from nnfs.layers import Dense, ReLU, Softmax
from nnfs.optim import SGD
from nnfs.utils import train_test_split

def one_hot(y, num_classes):
    """Convert integer labels to one-hot encoding."""
    import numpy as np
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

def main():
    print("🔢 MNIST MLP Example (Synthetic Data)")
    print("=" * 50)
    
    # Set backend
    print(f"Backend: {nnfs.get_backend_name()}")
    
    # Generate synthetic MNIST-like data
    import numpy as np
    np.random.seed(42)
    
    # Simulate MNIST: 28x28 grayscale images, 10 classes
    n_samples = 2000
    n_features = 28 * 28
    n_classes = 10
    
    X_data = np.random.randn(n_samples, n_features).astype(np.float32)
    y_idx = np.random.randint(0, n_classes, size=(n_samples,))
    y_data = one_hot(y_idx, n_classes)
    
    X = nnfs.Tensor(X_data)
    y = nnfs.Tensor(y_data)
    
    print(f"Dataset shape: {X.shape()}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {np.bincount(y_idx)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build MLP model
    model = nnfs.Sequential(
        Dense(n_features, 256),
        ReLU(),
        Dense(256, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, n_classes),
        Softmax()
    )
    
    print("\nModel Architecture:")
    nnfs.print_model_summary(model, input_shape=(n_features,))
    
    # Setup training
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nnfs.CrossEntropyLoss()
    
    config = nnfs.TrainerConfig(
        epochs=30,
        batch_size=64,
        learning_rate=0.01,
        log_interval=5
    )
    
    trainer = nnfs.Trainer(model, loss_fn, optimizer, config)
    
    # Train
    print("\n🚀 Training...")
    history = trainer.fit(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    train_loss, train_acc = trainer.evaluate(X_train, y_train)
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Test predictions
    print("\n🔮 Sample Predictions:")
    model.eval()
    
    # Get predictions for first 10 test samples
    sample_x = X_test[:10]
    sample_pred = model(sample_x)
    pred_classes = sample_pred.data.argmax(axis=1)
    true_classes = y_test[:10].data.argmax(axis=1)
    
    for i in range(10):
        print(f"Sample {i+1}: Predicted {pred_classes[i]}, True {true_classes[i]}")

if __name__ == '__main__':
    main()
