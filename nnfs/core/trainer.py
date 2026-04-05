"""Trainer system for neural network training."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from .backend import get_backend_name
from .tensor import Tensor
from ..utils.metrics import accuracy_score


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    debug_mode: bool = False
    gradient_explosion_threshold: float = 100.0
    log_interval: int = 10
    save_history: bool = True
    early_stopping_patience: Optional[int] = None
    validate_every: int = 1


@dataclass
class TrainingHistory:
    """Training history data."""
    train_loss: List[float]
    train_accuracy: List[float]
    val_loss: List[float]
    val_accuracy: List[float]
    epoch_times: List[float]
    learning_rates: List[float]


class Trainer:
    """Training system for neural networks."""
    
    def __init__(
        self,
        model: Module,
        loss_fn: Any,
        optimizer: Any,
        config: Optional[TrainerConfig] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config or TrainerConfig()
        
        self.history = TrainingHistory(
            train_loss=[],
            train_accuracy=[],
            val_loss=[],
            val_accuracy=[],
            epoch_times=[],
            learning_rates=[]
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)
    
    def _check_gradient_explosion(self) -> bool:
        """Check for gradient explosion."""
        import numpy as np
        for param in self.model.parameters():
            if param.grad is not None:
                grad_array = np.array(param.grad.data)
                grad_norm = float(grad_array.max())
                if grad_norm > self.config.gradient_explosion_threshold:
                    if self.config.debug_mode:
                        print(f"⚠️  Gradient explosion detected: {grad_norm:.2f}")
                    return True
        return False
    
    def _log_metrics(self, epoch: int, train_loss: float, train_acc: float, 
                    val_loss: Optional[float] = None, val_acc: Optional[float] = None) -> None:
        """Log training metrics."""
        if epoch % self.config.log_interval == 0:
            msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            msg += f" | LR: {self.optimizer.get_learning_rate():.6f}"
            print(msg)
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if self.config.early_stopping_patience is None:
            return False
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
    
    def _run_epoch(self, X_train: Tensor, y_train: Tensor, 
                   X_val: Optional[Tensor] = None, y_val: Optional[Tensor] = None,
                   epoch: int = 1) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Run a single training epoch."""
        self.model.train()
        epoch_start_time = time.time()
        
        # Training
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Simple batching (can be improved with proper data loader)
        n_samples = X_train.shape[0]
        for i in range(0, n_samples, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, n_samples)
            
            # Get batch
            x_batch = Tensor(X_train.data[i:batch_end])
            y_batch = Tensor(y_train.data[i:batch_end])
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x_batch)
            loss = self.loss_fn(output, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Check for gradient explosion
            if self._check_gradient_explosion():
                if self.config.debug_mode:
                    print("⚠️  Skipping update due to gradient explosion")
                continue
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * (batch_end - i)
            
            # Calculate accuracy (for classification)
            if hasattr(y_batch, 'shape') and len(y_batch.shape) == 1:
                # Binary classification
                predictions = (output.data > 0.5).astype(int)
                train_correct += (predictions.flatten() == y_batch.data.flatten()).sum()
            else:
                # Multi-class classification
                pred_classes = output.data.argmax(axis=1)
                true_classes = y_batch.data.argmax(axis=1)
                train_correct += (pred_classes == true_classes).sum()
            
            train_total += batch_end - i
        
        train_loss /= train_total if train_total > 0 else 1
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Validation
        val_loss = None
        val_accuracy = None
        if X_val is not None and y_val is not None and epoch % self.config.validate_every == 0:
            val_loss, val_accuracy = self.evaluate(X_val, y_val)
        
        epoch_time = time.time() - epoch_start_time
        
        return train_loss, train_accuracy, val_loss, val_accuracy
    
    def fit(self, X_train: Tensor, y_train: Tensor, 
            X_val: Optional[Tensor] = None, y_val: Optional[Tensor] = None) -> TrainingHistory:
        """Train the model."""
        print(f"🚀 Starting training for {self.config.epochs} epochs")
        print(f"📊 Batch size: {self.config.batch_size} | Learning rate: {self.config.learning_rate}")
        print(f"🔧 Backend: {get_backend_name()}")
        print("-" * 80)
        
        for epoch in range(1, self.config.epochs + 1):
            # Run epoch
            train_loss, train_acc, val_loss, val_acc = self._run_epoch(X_train, y_train, X_val, y_val, epoch)
            
            # Record history
            self.history.train_loss.append(train_loss)
            self.history.train_accuracy.append(train_acc)
            if val_loss is not None:
                self.history.val_loss.append(val_loss)
                self.history.val_accuracy.append(val_acc)
            self.history.learning_rates.append(self.optimizer.get_learning_rate())
            
            # Log metrics
            self._log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Check early stopping
            if val_loss is not None and self._should_early_stop(val_loss):
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break
            
            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch, train_loss, train_acc, val_loss, val_acc)
        
        print("-" * 80)
        print("✅ Training completed!")
        
        # Save history if requested
        if self.config.save_history:
            self.save_history("training_history.json")
        
        return self.history
    
    def evaluate(self, X: Tensor, y: Tensor) -> Tuple[float, float]:
        """Evaluate the model on validation data."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        n_samples = X.shape[0]
        for i in range(0, n_samples, self.config.batch_size):
            batch_end = min(i + self.config.batch_size, n_samples)
            
            x_batch = Tensor(X.data[i:batch_end])
            y_batch = Tensor(y.data[i:batch_end])
            
            # Forward pass
            output = self.model(x_batch)
            loss = self.loss_fn(output, y_batch)
            
            total_loss += loss.item() * (batch_end - i)
            
            # Handle different label formats
            if hasattr(y_batch, 'shape') and len(y_batch.shape) == 1:
                # Binary classification
                predictions = (output.data > 0.5).astype(int)
                correct += (predictions.flatten() == y_batch.data.flatten()).sum()
            else:
                # Multi-class classification
                pred_classes = output.data.argmax(axis=1)
                true_classes = y_batch.data.argmax(axis=1)
                correct += (pred_classes == true_classes).sum()
            
            total += batch_end - i
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, X: Tensor) -> Tensor:
        """Make predictions."""
        self.model.eval()
        return self.model(X)
    
    def save_history(self, filepath: str) -> None:
        """Save training history to JSON file."""
        history_dict = asdict(self.history)
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"💾 Training history saved to {filepath}")
    
    def load_history(self, filepath: str) -> None:
        """Load training history from JSON file."""
        with open(filepath, 'r') as f:
            history_dict = json.load(f)
        
        self.history = TrainingHistory(**history_dict)
        print(f"📂 Training history loaded from {filepath}")
    
    def save_model(self, filepath: str) -> None:
        """Save model state dict."""
        state_dict = self.model.state_dict()
        with open(filepath, 'w') as f:
            json.dump({k: v.tolist() for k, v in state_dict.items()}, f, indent=2)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model state dict."""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Convert lists back to arrays
        import numpy as np
        state_dict = {k: np.array(v) for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        print(f"📂 Model loaded from {filepath}")


# Callback functions
class EarlyStoppingCallback:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
    
    def __call__(self, trainer: Trainer, epoch: int, train_loss: float, train_acc: float, 
                 val_loss: Optional[float], val_acc: Optional[float]) -> None:
        if val_loss is None:
            return
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                trainer.should_stop = True


class LoggingCallback:
    """Logging callback for detailed metrics."""
    
    def __init__(self, log_file: str = "training.log"):
        self.log_file = log_file
    
    def __call__(self, trainer: Trainer, epoch: int, train_loss: float, train_acc: float, 
                 val_loss: Optional[float], val_acc: Optional[float]) -> None:
        with open(self.log_file, 'a') as f:
            log_entry = f"Epoch {epoch}: train_loss={train_loss:.6f}, train_acc={train_acc:.6f}"
            if val_loss is not None:
                log_entry += f", val_loss={val_loss:.6f}, val_acc={val_acc:.6f}"
            log_entry += f", lr={trainer.optimizer.get_learning_rate():.6f}\n"
            f.write(log_entry)
