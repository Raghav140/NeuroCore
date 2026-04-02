"""Container modules."""

from __future__ import annotations

from typing import List

from .module import Module


class Sequential(Module):
    """Apply child modules in sequence."""

    def __init__(self, *modules: Module):
        super().__init__()
        self._order: List[str] = []
        for idx, module in enumerate(modules):
            name = str(idx)
            self.add_module(name, module)
            self._order.append(name)

    def forward(self, x):
        out = x
        for name in self._order:
            out = self._modules[name](out)
        return out

    def backward(self, grad):
        # Autograd-first path: propagate from container output tensor.
        if not hasattr(self, "_last_output"):
            raise RuntimeError("No forward output cached; call model(input) before backward.")
        out = self._last_output
        if hasattr(out, "backward"):
            out.backward(grad)
            self._run_backward_hooks(grad, grad)
            return grad
        raise RuntimeError("Sequential backward requires Tensor output.")
