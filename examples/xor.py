"""XOR problem example using NNFS."""

import nnfs
from nnfs.layers import Dense, ReLU, Sigmoid
from nnfs.optim import SGD
from nnfs.utils import make_xor, train_test_split, accuracy_score

def main():
    print("🧠 XOR Problem Example")
    print("=" * 40)
    
    # Set backend
    print(f"Backend: {nnfs.get_backend_name()}")
    
    # Generate XOR data
    X, y = make_xor(n_samples=500, noise=0.1, random_state=42)
    print(f"Dataset shape: {X.shape()}")
    print(f"Labels: {y.shape()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = nnfs.Sequential(
        Dense(2, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 1),
        Sigmoid()
    )
    
    print("\nModel Architecture:")
    nnfs.print_model_summary(model, input_shape=(2,))
    
    # Setup training
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nnfs.BCELoss()
    
    config = nnfs.TrainerConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.1,
        log_interval=20
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
    test_points = [
        [-1, -1],  # Expected: 0
        [-1, 1],   # Expected: 1
        [1, -1],   # Expected: 1
        [1, 1]     # Expected: 0
    ]
    
    for point in test_points:
        x = nnfs.Tensor([point])
        pred = model(x).item()
        label = int(pred > 0.5)
        print(f"Input: {point} -> Prediction: {pred:.4f} -> Label: {label}")

if __name__ == '__main__':
    main()
