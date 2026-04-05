"""Loss function implementations."""

from __future__ import annotations

from ..core.module import Module
from ..core.tensor import Tensor
from ..core.backend import clip, sum, mean, get_backend


class MSELoss(Module):
    """Mean Squared Error loss."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSE loss."""
        diff = predictions - targets
        return (diff * diff).mean()
    
    def __repr__(self) -> str:
        return "MSELoss()"


class BCELoss(Module):
    """Binary Cross Entropy loss."""
    
    def __init__(self, epsilon: float = 1e-15):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute binary cross entropy loss."""
        # Clamp predictions to avoid log(0)
        eps = self.epsilon
        
        # Use element-wise operations that preserve gradients
        eps_tensor = Tensor(eps)
        one_minus_eps = Tensor(1.0 - eps)
        
        # Clamp predictions
        lower_clamped = predictions.maximum(eps_tensor)
        predictions_clamped = lower_clamped.minimum(one_minus_eps)
        
        # Ensure targets have the same shape
        if len(targets.shape) == 1 and len(predictions.shape) == 2:
            targets = targets.reshape((-1, 1))
        
        # Compute BCE loss elementwise
        # BCE = -(y * log(p) + (1-y) * log(1-p))
        term1 = targets * predictions_clamped.log()
        
        # Create ones with proper shapes
        ones_like_targets = Tensor(get_backend().xp.ones(targets.shape))
        ones_like_predictions = Tensor(get_backend().xp.ones(predictions.shape))
        
        term2 = (ones_like_targets - targets) * (ones_like_predictions - predictions_clamped).log()
        
        loss = -(term1 + term2)
        return loss.mean()
    
    def __repr__(self) -> str:
        return f"BCELoss(epsilon={self.epsilon})"


class CrossEntropyLoss(Module):
    """Cross Entropy loss for multi-class classification."""
    
    def __init__(self, epsilon: float = 1e-15):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute cross entropy loss."""
        # Apply softmax to get probabilities
        max_vals = predictions.max(axis=1, keepdims=True)
        exp_preds = (predictions - max_vals).exp()
        sum_exp = exp_preds.sum(axis=1, keepdims=True)
        probs = exp_preds / sum_exp
        
        # Clamp probabilities to avoid log(0)
        # Use a custom clamp implementation that preserves gradients
        eps = self.epsilon
        probs_clamped = Tensor(
            get_backend().xp.minimum(get_backend().xp.maximum(probs.data, eps), 1.0 - eps),
            requires_grad=probs.requires_grad,
            _children=(probs,),
            _op="clamp"
        )
        
        def _backward() -> None:
            if probs.requires_grad:
                if probs.grad is not None:
                    # Gradient of clamp is 1 for values within bounds, 0 otherwise
                    mask = ((probs.data > eps) & (probs.data < 1.0 - eps)).astype(float)
                    probs.grad.data += mask * probs_clamped.grad.data
        
        probs_clamped._backward = _backward
        
        # Cross entropy: -sum(y_i * log(p_i))
        # targets are assumed to be one-hot encoded
        loss = -(targets * probs_clamped.log()).sum(axis=1)
        return loss.mean()
    
    def __repr__(self) -> str:
        return f"CrossEntropyLoss(epsilon={self.epsilon})"
