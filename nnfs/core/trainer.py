"""Training abstraction with callbacks and logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .backend import asarray, to_numpy, xp


class Callback:
    """Trainer callback base class."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict) -> None:
        pass


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str = "val_loss"
    debug_mode: bool = False
    gradient_explosion_threshold: float = 1e3


class Trainer:
    """Reusable trainer for Module models."""

    def __init__(self, model, loss_fn, optimizer, scheduler=None, callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.history = {
            "loss": [],
            "val_loss": [],
            "lr": [],
            "accuracy": [],
            "val_accuracy": [],
            "grad_norm": [],
            "stopped_epoch": None,
        }

    def _accuracy(self, y_true, y_pred) -> float:
        y_true_np = to_numpy(y_true)
        y_pred_np = to_numpy(y_pred)
        if y_pred_np.ndim > 1 and y_pred_np.shape[1] > 1:
            pred = y_pred_np.argmax(axis=1)
            truth = y_true_np.argmax(axis=1) if y_true_np.ndim > 1 else y_true_np.reshape(-1)
            return float((pred == truth).mean())
        pred = (y_pred_np.reshape(-1) >= 0.5).astype(int)
        truth = y_true_np.reshape(-1).astype(int)
        return float((pred == truth).mean())

    def fit(self, X, y, config: TrainerConfig, X_val=None, y_val=None):
        X = asarray(X)
        y = asarray(y)
        if X_val is not None:
            X_val = asarray(X_val)
        if y_val is not None:
            y_val = asarray(y_val)
        n = X.shape[0]
        best_metric = None
        bad_epochs = 0
        xp_mod = xp()
        for epoch in range(1, config.epochs + 1):
            self.model.train()
            if self.scheduler is not None and hasattr(self.scheduler, "step_epoch_start"):
                self.scheduler.step_epoch_start(epoch)
            lr = self.optimizer.lr

            idx = xp_mod.arange(n)
            xp_mod.random.shuffle(idx)
            loss_sum = 0.0
            acc_sum = 0.0
            batches = 0

            for start in range(0, n, config.batch_size):
                batch_idx = idx[start : start + config.batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]

                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn.forward(preds, yb)
                if hasattr(loss, "backward"):
                    loss.backward()
                    loss_value = float(to_numpy(loss.data).reshape(-1)[0])
                else:
                    grad = self.loss_fn.backward(preds, yb)
                    self.model.backward(grad)
                    loss_value = float(loss)

                grad_norm_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm_sq += float((to_numpy(p.grad) ** 2).sum())
                grad_norm = grad_norm_sq**0.5
                if config.debug_mode:
                    if not xp_mod.isfinite(grad_norm):
                        raise RuntimeError("Gradient became NaN/Inf in debug mode.")
                    if grad_norm > config.gradient_explosion_threshold:
                        raise RuntimeError(
                            f"Gradient explosion detected: norm={grad_norm:.4f} > "
                            f"{config.gradient_explosion_threshold:.4f}"
                        )
                self.optimizer.step()

                loss_sum += loss_value
                acc_sum += self._accuracy(yb, preds)
                batches += 1

            train_loss = loss_sum / max(1, batches)
            train_acc = acc_sum / max(1, batches)

            self.model.eval()
            val_loss = None
            val_acc = None
            if X_val is not None and y_val is not None:
                val_pred = self.model(X_val)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                val_acc = self._accuracy(y_val, val_pred)

            if self.scheduler is not None and hasattr(self.scheduler, "step_metric"):
                monitor_val = val_loss if val_loss is not None else train_loss
                lr = self.scheduler.step_metric(monitor_val)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["lr"].append(lr)
            self.history["grad_norm"].append(grad_norm)
            if val_loss is not None:
                self.history["val_loss"].append(val_loss)
            if val_acc is not None:
                self.history["val_accuracy"].append(val_acc)

            logs = {
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": lr,
            }
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if config.early_stopping_patience is not None:
                if config.early_stopping_monitor == "val_loss" and val_loss is not None:
                    metric = val_loss
                else:
                    metric = train_loss
                if best_metric is None or metric < (best_metric - config.early_stopping_min_delta):
                    best_metric = metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= config.early_stopping_patience:
                        self.history["stopped_epoch"] = epoch
                        break
        return self.history

    def save_history_json(self, path: str) -> None:
        """Persist training history as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
