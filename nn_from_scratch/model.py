import json
from typing import List, Optional

import numpy as np

from .activations import ReLU, Sigmoid, Softmax, Tanh
from .layers import Dense, Dropout, Layer
from .losses import Loss
from .optimizer import Optimizer


class NeuralNetwork:
    """
    Simple feedforward neural network supporting multiple layers,
    backpropagation, and gradient-based optimization.
    """

    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_fn: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None

    def add(self, layer: Layer) -> None:
        """Append a layer to the model."""
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Configure the model with loss function and optimizer.
        """
        self.loss_fn = loss
        self.optimizer = optimizer

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagate through all layers.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward propagate the gradient through all layers.
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            # Many activation and dropout layers will simply propagate the gradient.
            grad = layer.backward(grad)

    def _update_params(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call compile() first.")
        self.optimizer.step(self.layers)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        scheduler=None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        early_stopping_monitor: str = "val_loss",
    ):
        """
        Train the network with optional scheduler and early stopping.
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("Model is not compiled. Call compile() first.")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or None.")

        n_samples = X.shape[0]

        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples

        history = {"loss": [], "val_loss": [], "lr": [], "stopped_epoch": None}
        best_metric = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            if scheduler is not None and hasattr(scheduler, "step_epoch_start"):
                current_lr = scheduler.step_epoch_start(epoch)
            else:
                current_lr = self.optimizer.lr

            # Shuffle indices for each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_idx = indices[start_idx:end_idx]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward
                preds = self.forward(X_batch, training=True)
                loss = self.loss_fn.forward(y_batch, preds)
                epoch_loss += loss
                num_batches += 1

                # Backward
                loss_grad = self.loss_fn.backward(y_batch, preds)
                self.backward(loss_grad)

                # Parameter update
                self._update_params()

            avg_loss = epoch_loss / max(1, num_batches)
            history["loss"].append(avg_loss)
            history["lr"].append(current_lr)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_preds = self.forward(X_val, training=False)
                val_loss = self.loss_fn.forward(y_val, val_preds)
                history["val_loss"].append(val_loss)

            if scheduler is not None and hasattr(scheduler, "step_metric"):
                monitored = val_loss if val_loss is not None else avg_loss
                current_lr = scheduler.step_metric(monitored)
                history["lr"][-1] = current_lr

            if verbose:
                msg = f"Epoch {epoch:03d} - loss: {avg_loss:.4f} - lr: {current_lr:.6f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)

            if early_stopping_patience is not None:
                if early_stopping_monitor == "val_loss" and val_loss is not None:
                    monitored_metric = val_loss
                else:
                    monitored_metric = avg_loss

                if best_metric is None or monitored_metric < (
                    best_metric - early_stopping_min_delta
                ):
                    best_metric = monitored_metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stopping_patience:
                        history["stopped_epoch"] = epoch
                        if verbose:
                            print(f"Early stopping at epoch {epoch:03d}")
                        break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on new data.
        """
        return self.forward(X, training=False)

    def save(self, path_prefix: str) -> None:
        """
        Save architecture and learned parameters.

        Writes:
        - `<path_prefix>.json` for architecture metadata
        - `<path_prefix>.npz` for numeric tensors
        """
        layer_configs = []
        tensors = {}
        dense_count = 0

        for layer in self.layers:
            layer_name = layer.__class__.__name__
            if isinstance(layer, Dense):
                layer_configs.append(
                    {
                        "type": "Dense",
                        "input_dim": layer.input_dim,
                        "output_dim": layer.output_dim,
                        "l2_lambda": layer.l2_lambda,
                    }
                )
                tensors[f"dense_{dense_count}_W"] = layer.W
                tensors[f"dense_{dense_count}_b"] = layer.b
                dense_count += 1
            elif isinstance(layer, Dropout):
                layer_configs.append({"type": "Dropout", "rate": layer.rate})
            elif layer_name in {"ReLU", "Sigmoid", "Tanh", "Softmax"}:
                layer_configs.append({"type": layer_name})
            else:
                raise ValueError(f"Unsupported layer for save: {layer_name}")

        metadata = {"layers": layer_configs}
        with open(f"{path_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        np.savez(f"{path_prefix}.npz", **tensors)

    @classmethod
    def load(cls, path_prefix: str):
        """
        Load architecture and learned parameters into a new model.
        """
        with open(f"{path_prefix}.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        tensors = np.load(f"{path_prefix}.npz")

        model = cls()
        dense_count = 0
        activation_map = {
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "Tanh": Tanh,
            "Softmax": Softmax,
        }

        for layer_cfg in metadata["layers"]:
            layer_type = layer_cfg["type"]
            if layer_type == "Dense":
                layer = Dense(
                    input_dim=layer_cfg["input_dim"],
                    output_dim=layer_cfg["output_dim"],
                    weight_init="xavier",
                    l2_lambda=layer_cfg.get("l2_lambda", 0.0),
                )
                layer.W = tensors[f"dense_{dense_count}_W"]
                layer.b = tensors[f"dense_{dense_count}_b"]
                dense_count += 1
            elif layer_type == "Dropout":
                layer = Dropout(rate=layer_cfg["rate"])
            elif layer_type in activation_map:
                layer = activation_map[layer_type]()
            else:
                raise ValueError(f"Unsupported layer type in saved file: {layer_type}")
            model.add(layer)

        return model

