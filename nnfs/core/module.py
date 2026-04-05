"""Base Module class for neural network components."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple, Union
from collections import OrderedDict

from .parameter import Parameter
from .tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._modules: Dict[str, Module] = OrderedDict()
        self._buffers: Dict[str, Tensor] = OrderedDict()
        self._backward_hooks: Dict[str, List[callable]] = OrderedDict()
        self._forward_hooks: Dict[str, List[callable]] = OrderedDict()
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override to automatically register parameters and submodules."""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif name.startswith('_') or not hasattr(self, '_buffers'):
            super().__setattr__(name, value)
        else:
            # Handle buffers (non-parameter tensors)
            if hasattr(value, 'data'):  # It's a Tensor
                self._buffers[name] = value
            super().__setattr__(name, value)
    
    def __getattr__(self, name: str) -> Any:
        """Override to access submodules, parameters, and buffers."""
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        elif '_modules' in self.__dict__ and name in self._modules:
            return self._modules[name]
        elif '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __delattr__(self, name: str) -> None:
        """Override to unregister parameters and submodules."""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make module callable."""
        # Forward pre-hooks
        for hook in self._forward_hooks.get('pre_forward', []):
            hook(self, args, kwargs)
        
        result = self.forward(*args, **kwargs)
        
        # Forward post-hooks
        for hook in self._forward_hooks.get('post_forward', []):
            hook(self, args, kwargs, result)
        
        return result
    
    def train(self) -> None:
        """Set module to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self) -> None:
        """Set module to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters, yielding both name and parameter."""
        for name, param in self._parameters.items():
            yield prefix + name, param
        if recurse:
            for name, module in self._modules.items():
                yield from module.named_parameters(prefix + name + '.', recurse=True)
    
    def zero_grad(self) -> None:
        """Reset gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()
    
    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a buffer (non-parameter tensor)."""
        setattr(self, name, tensor)
    
    def add_module(self, name: str, module: Module) -> None:
        """Add a submodule to this module."""
        setattr(self, name, module)
    
    def state_dict(self, prefix: str = '') -> Dict[str, Tensor]:
        """Return a dictionary containing a whole state of the module."""
        state = {}
        for name, param in self._parameters.items():
            state[prefix + name] = param.data
        for name, module in self._modules.items():
            state.update(module.state_dict(prefix + name + '.'))
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Copy parameters and buffers from state_dict into this module."""
        current_state = self.state_dict()
        
        if strict:
            missing_keys = set(current_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(current_state.keys())
            
            if missing_keys:
                raise KeyError(f"Missing key(s) in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected key(s) in state_dict: {unexpected_keys}")
        
        # Load parameters
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]
    
    def apply(self, fn: callable) -> Module:
        """Apply a function recursively to every submodule."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    
    def children(self) -> Iterator[Module]:
        """Return an iterator over immediate child modules."""
        return self._modules.values()
    
    def modules(self) -> Iterator[Module]:
        """Return an iterator over all modules in the network."""
        yield self
        for module in self.children():
            yield from module.modules()
    
    def register_backward_hook(self, hook: callable) -> None:
        """Register a backward hook."""
        if 'backward' not in self._backward_hooks:
            self._backward_hooks['backward'] = []
        self._backward_hooks['backward'].append(hook)
    
    def register_forward_pre_hook(self, hook: callable) -> None:
        """Register a forward pre-hook."""
        if 'pre_forward' not in self._forward_hooks:
            self._forward_hooks['pre_forward'] = []
        self._forward_hooks['pre_forward'].append(hook)
    
    def register_forward_hook(self, hook: callable) -> None:
        """Register a forward post-hook."""
        if 'post_forward' not in self._forward_hooks:
            self._forward_hooks['post_forward'] = []
        self._forward_hooks['post_forward'].append(hook)
    
    def __repr__(self) -> str:
        """String representation of the module."""
        class_name = self.__class__.__name__
        lines = [class_name + '(']
        
        for name, module in self._modules.items():
            module_repr = repr(module).split('\n')
            lines.append(f'  ({name}): {module_repr[0]}')
            for line in module_repr[1:]:
                lines.append(f'    {line}')
        
        for name, param in self._parameters.items():
            lines.append(f'  ({name}): Parameter({param.data.shape})')
        
        if len(lines) == 1:
            lines[0] += ')'
        else:
            lines.append(')')
        
        return '\n'.join(lines)


class Sequential(Module):
    """A sequential container that passes input through a sequence of modules."""
    
    def __init__(self, *args: Union[Module, Tuple[str, Module]]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        
        for idx, module in enumerate(args):
            if isinstance(module, tuple):
                name, module = module
            else:
                name = str(idx)
            self.add_module(name, module)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'Sequential']:
        if isinstance(idx, slice):
            return Sequential(*list(self._modules.values())[idx])
        else:
            return list(self._modules.values())[idx]
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
