"""Binary classification example using NNFS."""

import nnfs
from nnfs.layers import Dense, ReLU, Sigmoid
from nnfs.optim import SGD
from nnfs.utils import make_binary_classification, train_test_split, accuracy_score

def main():
    print("🎯 Binary Classification Example")
    print("=" * 50)
    
    # Set backend
    print(f"Backend: {nnfs.get_backend_name()}")
    
    # Generate synthetic data
    X, y = make_binary_classification(n_samples=1000, n_features=2, noise=0.2, random_state=42)
    print(f"Dataset shape: {X.shape()}")
    print(f"Class distribution: {int(y.data.sum())}/{len(y.data)} positive/negative")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = nnfs.Sequential(
        Dense(2, 32),
        ReLU(),
        Dense(32, 16),
        ReLU(),
        Dense(16, 1),
        Sigmoid()
    )
    
    print("\nModel Architecture:")
    nnfs.print_model_summary(model, input_shape=(2,))
    
    # Setup training
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nnfs.BCELoss()
    
    config = nnfs.TrainerConfig(
        epochs=150,
        batch_size=32,
        learning_rate=0.05,
        log_interval=15
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
    
    # Detailed metrics
    print("\n📈 Detailed Metrics:")
    model.eval()
    y_pred_prob = model(X_test)
    y_pred = nnfs.Tensor((y_pred_prob.data > 0.5).astype(int))
    
    print(nnfs.classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
