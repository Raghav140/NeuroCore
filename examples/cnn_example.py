"""CNN example using NNFS."""

import nnfs
from nnfs.layers import Conv2D, Dense, Flatten, MaxPooling2D, ReLU, Sigmoid
from nnfs.optim import SGD
from nnfs.utils import train_test_split

def main():
    print("🖼️  CNN Example")
    print("=" * 40)
    
    # Set backend
    print(f"Backend: {nnfs.get_backend_name()}")
    
    # Generate synthetic image data
    import numpy as np
    np.random.seed(42)
    
    # Simulate small grayscale images: 1 channel, 8x8 pixels
    n_samples = 500
    height, width = 8, 8
    channels = 1
    
    X_data = np.random.randn(n_samples, channels, height, width).astype(np.float32)
    y_data = (np.random.rand(n_samples, 1) > 0.5).astype(np.float32)
    
    X = nnfs.Tensor(X_data)
    y = nnfs.Tensor(y_data)
    
    print(f"Dataset shape: {X.shape()}")
    print(f"Image size: {height}x{height}, Channels: {channels}")
    print(f"Class distribution: {int(y.data.sum())}/{len(y.data)} positive/negative")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build CNN model
    model = nnfs.Sequential(
        Conv2D(channels, 4, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPooling2D(kernel_size=2),
        Conv2D(4, 8, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPooling2D(kernel_size=2),
        Flatten(),
        Dense(8 * 2 * 2, 16),
        ReLU(),
        Dense(16, 1),
        Sigmoid()
    )
    
    print("\nModel Architecture:")
    nnfs.print_model_summary(model, input_shape=(channels, height, width))
    
    # Setup training
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nnfs.BCELoss()
    
    config = nnfs.TrainerConfig(
        epochs=20,
        batch_size=32,
        learning_rate=0.01,
        debug_mode=True,
        log_interval=3
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
    
    # Get predictions for first 5 test samples
    sample_x = X_test[:5]
    sample_pred = model(sample_x)
    
    for i in range(5):
        pred_prob = sample_pred[i].item()
        pred_label = int(pred_prob > 0.5)
        true_label = int(y_test[i].item())
        print(f"Sample {i+1}: Pred={pred_prob:.4f} ({pred_label}), True={true_label}")

if __name__ == '__main__':
    main()
