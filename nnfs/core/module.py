"""Core Module and Parameter abstractions."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .tensor import Tensor


class Parameter(Tensor):
    """A trainable Tensor parameter."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """PyTorch-style base class for all layers/models."""

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_forward_hooks", [])
        object.__setattr__(self, "_backward_hooks", [])
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        object.__setattr__(self, name, value)

    def add_module(self, name: str, module: "Module") -> None:
        """Register child module."""
        setattr(self, name, module)

    def register_parameter(self, name: str, param: Parameter) -> None:
        """Register trainable parameter."""
        setattr(self, name, param)

    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass."""
        raise NotImplementedError

    def __call__(self, x):
        out = self.forward(x)
        self._last_output = out
        for hook in self._forward_hooks:
            hook(self, x, out)
        return out

    def parameters(self) -> Iterator[Parameter]:
        """Iterate all parameters recursively."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """Iterate all named parameters recursively."""
        for name, param in self._parameters.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            yield full, param
        for mod_name, module in self._modules.items():
            mod_prefix = f"{prefix}{mod_name}" if not prefix else f"{prefix}.{mod_name}"
            yield from module.named_parameters(mod_prefix)

    def modules(self) -> Iterator["Module"]:
        """Yield self and all child modules recursively."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def zero_grad(self) -> None:
        """Zero all parameter gradients recursively."""
        for param in self.parameters():
            param.zero_grad()

    def train(self) -> None:
        """Switch module and submodules to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        """Switch module and submodules to eval mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()

    def register_forward_hook(self, hook: Callable) -> None:
        """Register a forward hook: hook(module, inp, out)."""
        self._forward_hooks.append(hook)

    def register_backward_hook(self, hook: Callable) -> None:
        """Register a backward hook: hook(module, grad_in, grad_out)."""
        self._backward_hooks.append(hook)

    def _run_backward_hooks(self, grad_in, grad_out) -> None:
        for hook in self._backward_hooks:
            hook(self, grad_in, grad_out)

    def backward(self, grad=None):
        """Backprop from latest output tensor."""
        if not hasattr(self, "_last_output"):
            raise RuntimeError("No forward output cached; call model(input) before backward.")
        out = self._last_output
        if not isinstance(out, Tensor):
            raise RuntimeError("Autograd backward expects Tensor output.")
        out.backward(grad)

    def state_dict(self) -> Dict[str, object]:
        """Return parameter snapshot dictionary."""
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Load parameter snapshot dictionary."""
        named = dict(self.named_parameters())
        for name, data in state.items():
            if name not in named:
                continue
            named[name].data[...] = data
