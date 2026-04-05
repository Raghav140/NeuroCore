"""Model summary and visualization utilities."""

from __future__ import annotations

from typing import List, Dict, Any

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor import Tensor
from ..core.backend import get_backend


def count_parameters(model: Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.data.size
        if param.requires_grad:
            trainable_params += param.data.size
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_info(model: Module, input_shape: tuple = None) -> Dict[str, Any]:
    """Get detailed model information."""
    info = {
        'model_type': model.__class__.__name__,
        'parameters': count_parameters(model),
        'layers': [],
        'submodules': []
    }
    
    # Add input/output shapes if provided
    if input_shape:
        dummy_input = Tensor(get_backend().xp.random.randn(*input_shape))
        try:
            output = model(dummy_input)
            info['input_shape'] = input_shape
            info['output_shape'] = output.shape
        except:
            info['input_shape'] = input_shape
            info['output_shape'] = 'Unknown (forward failed)'
    
    # Collect layer information
    def collect_layer_info(module: Module, prefix: str = '') -> None:
        for name, submodule in module._modules.items():
            layer_info = {
                'name': prefix + name,
                'type': submodule.__class__.__name__,
                'parameters': count_parameters(submodule)
            }
            
            # Add specific layer info
            if hasattr(submodule, 'in_features') and hasattr(submodule, 'out_features'):
                layer_info['in_features'] = submodule.in_features
                layer_info['out_features'] = submodule.out_features
            
            info['layers'].append(layer_info)
            
            # Recursively collect from submodules
            collect_layer_info(submodule, prefix + name + '.')
    
    collect_layer_info(model)
    
    return info


def model_summary(model: Module, input_shape: tuple = None, show_weights: bool = False) -> str:
    """Generate a detailed model summary."""
    info = get_model_info(model, input_shape)
    
    summary = []
    summary.append("=" * 80)
    summary.append(f"MODEL SUMMARY: {info['model_type']}")
    summary.append("=" * 80)
    
    if input_shape:
        summary.append(f"Input Shape: {info['input_shape']}")
        summary.append(f"Output Shape: {info['output_shape']}")
        summary.append("")
    
    # Parameters summary
    params = info['parameters']
    summary.append("Parameters:")
    summary.append(f"  Total:        {params['total']:,}")
    summary.append(f"  Trainable:   {params['trainable']:,}")
    summary.append(f"  Non-trainable: {params['non_trainable']:,}")
    summary.append("")
    
    # Layer summary
    summary.append("Layer Architecture:")
    summary.append("-" * 80)
    summary.append(f"{'Layer Name':<30} {'Type':<20} {'Input':<15} {'Output':<15} {'Params':<10}")
    summary.append("-" * 80)
    
    for layer in info['layers']:
        name = layer['name'][:29]
        layer_type = layer['type'][:19]
        
        # Try to get input/output shapes
        in_shape = str(getattr(layer, 'in_features', 'N/A'))[:14]
        out_shape = str(getattr(layer, 'out_features', 'N/A'))[:14]
        
        param_count = layer['parameters']['total']
        
        summary.append(f"{name:<30} {layer_type:<20} {in_shape:<15} {out_shape:<15} {param_count:<10,}")
    
    summary.append("-" * 80)
    summary.append("")
    
    # Weight statistics (if requested)
    if show_weights:
        summary.append("Weight Statistics:")
        summary.append("-" * 50)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                weight_mean = float(weight_data.mean())
                weight_std = float(weight_data.std())
                weight_min = float(weight_data.min())
                weight_max = float(weight_data.max())
                
                summary.append(f"{name}:")
                summary.append(f"  Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
                summary.append(f"  Min: {weight_min:.6f}, Max: {weight_max:.6f}")
                summary.append("")
    
    return "\n".join(summary)


def print_model_summary(model: Module, input_shape: tuple = None, show_weights: bool = False) -> None:
    """Print model summary to console."""
    print(model_summary(model, input_shape, show_weights))


def visualize_model_architecture(model: Module) -> str:
    """Create a simple text visualization of model architecture."""
    
    def get_module_tree(module: Module, indent: int = 0) -> List[str]:
        lines = []
        prefix = "  " * indent
        
        # Add current module
        module_name = module.__class__.__name__
        params = count_parameters(module)
        
        # Special formatting for different layer types
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            lines.append(f"{prefix}{module_name}({module.in_features} → {module.out_features}) [{params['total']:,}]")
        elif hasattr(module, 'kernel_size'):
            lines.append(f"{prefix}{module_name}(kernel={module.kernel_size}) [{params['total']:,}]")
        else:
            lines.append(f"{prefix}{module_name} [{params['total']:,}]")
        
        # Add submodules
        for name, submodule in module._modules.items():
            sub_lines = get_module_tree(submodule, indent + 1)
            lines.extend(sub_lines)
        
        return lines
    
    lines = get_module_tree(model)
    return "\n".join(lines)


def compare_models(models: Dict[str, Module], input_shape: tuple = None) -> str:
    """Compare multiple models side by side."""
    comparison = []
    comparison.append("=" * 100)
    comparison.append("MODEL COMPARISON")
    comparison.append("=" * 100)
    
    # Header
    header = f"{'Model':<20} {'Total Params':<15} {'Trainable':<12} {'Non-trainable':<15} {'Size (MB)':<12}"
    comparison.append(header)
    comparison.append("-" * 100)
    
    # Model info
    for name, model in models.items():
        params = count_parameters(model)
        
        # Estimate model size (assuming 4 bytes per parameter for float32)
        size_mb = params['total'] * 4 / (1024 * 1024)
        
        line = f"{name:<20} {params['total']:<15,} {params['trainable']:<12,} {params['non_trainable']:<15,} {size_mb:<12.2f}"
        comparison.append(line)
    
    comparison.append("-" * 100)
    comparison.append("")
    
    # Detailed comparison
    for name, model in models.items():
        comparison.append(f"{name}:")
        comparison.append(visualize_model_architecture(model))
        comparison.append("")
    
    return "\n".join(comparison)


def analyze_model_complexity(model: Module, input_shape: tuple) -> Dict[str, Any]:
    """Analyze model computational complexity."""
    dummy_input = Tensor(get_backend().xp.random.randn(*input_shape))
    
    # Count operations (rough estimate)
    total_ops = 0
    layer_ops = {}
    
    def analyze_module(module: Module, x: Tensor) -> Tensor:
        nonlocal total_ops
        
        module_name = module.__class__.__name__
        ops = 0
        
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            # Dense layer: in_features * out_features multiplications + out_features additions
            ops = module.in_features * module.out_features + module.out_features
        elif hasattr(module, 'kernel_size'):
            # Conv layer: rough estimate
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                ops = (module.kernel_size ** 2) * module.in_channels * module.out_channels
        
        if ops > 0:
            layer_ops[module_name] = layer_ops.get(module_name, 0) + ops
            total_ops += ops
        
        # Forward pass
        output = module(x)
        
        # Handle submodules
        for name, submodule in module._modules.items():
            output = analyze_module(submodule, output)
        
        return output
    
    try:
        _ = analyze_module(model, dummy_input)
    except:
        pass
    
    return {
        'total_operations': total_ops,
        'operations_by_layer': layer_ops,
        'parameters': count_parameters(model)
    }
