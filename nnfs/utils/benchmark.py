"""Benchmarking utilities."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Any, Optional

from ..core.tensor import Tensor
from ..core.module import Module
from ..core.backend import get_backend, set_backend, get_backend_name


class Benchmark:
    """Benchmark class for measuring performance."""
    
    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time."""
        self.end_time = time.time()
        if self.start_time is None:
            raise ValueError("Benchmark not started")
        return self.end_time - self.start_time
    
    def measure_forward(self, model: Module, x: Tensor, warmup: int = 10, runs: int = 100) -> Dict[str, float]:
        """Measure forward pass performance."""
        # Warmup
        model.eval()
        for _ in range(warmup):
            _ = model(x)
        
        # Measure
        times = []
        for _ in range(runs):
            start = time.time()
            _ = model(x)
            end = time.time()
            times.append(end - start)
        
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def measure_backward(self, model: Module, x: Tensor, y: Tensor, loss_fn: Any, warmup: int = 10, runs: int = 100) -> Dict[str, float]:
        """Measure backward pass performance."""
        model.eval()
        
        # Warmup
        for _ in range(warmup):
            model.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
        
        # Measure
        times = []
        for _ in range(runs):
            model.zero_grad()
            start = time.time()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            end = time.time()
            times.append(end - start)
        
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def measure_training_epoch(self, model: Module, x: Tensor, y: Tensor, loss_fn: Any, optimizer: Any, batch_size: int = 32) -> Dict[str, float]:
        """Measure training epoch performance."""
        model.train()
        
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        
        n_samples = x.shape[0]
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            # Get batch
            x_batch = Tensor(x.data[i:batch_end])
            y_batch = Tensor(y.data[i:batch_end])
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        end = time.time()
        
        return {
            'epoch_time': end - start,
            'avg_loss': total_loss / num_batches,
            'throughput': n_samples / (end - start),  # samples per second
            'num_batches': num_batches
        }
    
    def benchmark_backends(self, model: Module, x: Tensor, y: Tensor, loss_fn: Any, backends: List[str] = ['numpy']) -> Dict[str, Any]:
        """Benchmark different backends."""
        results = {}
        
        original_backend = get_backend_name()
        
        for backend in backends:
            try:
                # Set backend
                set_backend(backend)
                
                # Re-create tensors with new backend
                x_backend = Tensor(x.data)
                y_backend = Tensor(y.data)
                
                print(f"Benchmarking {backend} backend...")
                
                # Measure forward pass
                forward_stats = self.measure_forward(model, x_backend)
                
                # Measure backward pass
                backward_stats = self.measure_backward(model, x_backend, y_backend, loss_fn)
                
                results[backend] = {
                    'forward': forward_stats,
                    'backward': backward_stats,
                    'backend': backend,
                    'tensor_shape': x_backend.shape,
                    'model_parameters': sum(p.data.size for p in model.parameters())
                }
                
                print(f"  Forward: {forward_stats['mean_time']*1000:.2f}ms ± {forward_stats['std_time']*1000:.2f}ms")
                print(f"  Backward: {backward_stats['mean_time']*1000:.2f}ms ± {backward_stats['std_time']*1000:.2f}ms")
                
            except Exception as e:
                print(f"  Error with {backend} backend: {e}")
                results[backend] = {'error': str(e)}
        
        # Restore original backend
        set_backend(original_backend)
        
        return results
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Benchmark results saved to {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY: {self.name}")
        print(f"{'='*60}")
        
        if not self.results:
            print("No results to display.")
            return
        
        for key, value in self.results.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"  {sub_key}:")
                        for k, v in sub_value.items():
                            if isinstance(v, float):
                                print(f"    {k}: {v:.6f}")
                            else:
                                print(f"    {k}: {v}")
                    else:
                        if isinstance(sub_value, float):
                            print(f"  {sub_key}: {sub_value:.6f}")
                        else:
                            print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {value}")


def benchmark_model(model: Module, x: Tensor, y: Tensor, loss_fn: Any, optimizer: Any, 
                    backends=['numpy'], epochs=5):
    """Convenience function to benchmark a model."""
    benchmark = Benchmark("Model Benchmark")
    
    # Benchmark different backends
    backend_results = benchmark.benchmark_backends(model, x, y, loss_fn, optimizer, backends, epochs)
    
    # Additional measurements
    model.eval()
    
    # Model size
    total_params = sum(p.data.size for p in model.parameters())
    trainable_params = sum(p.data.size for p in model.parameters() if p.requires_grad)
    
    # Memory usage (approximate)
    param_memory = sum(p.data.nbytes for p in model.parameters())
    
    # Combine results
    results = {
        'backends': backend_results,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_bytes': param_memory
        }
    }
    
    return results


def benchmark_backends(model, X, y, loss_fn, optimizer, backends=['numpy'], epochs=5):
    """Benchmark model performance across different backends."""
    from ..core.backend import set_backend, get_backend_name
    import time
    import numpy as np
    
    results = {}
    
    for backend in backends:
        print(f"Benchmarking {backend} backend...")
        
        # Set backend
        set_backend(backend)
        
        # Create fresh model copy for this backend
        # Note: This is a simplified version - in practice you'd want to properly clone the model
        model_copy = model  # For now, reuse the same model
        
        # Warmup
        for _ in range(3):
            output = model_copy(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Benchmark forward pass
        forward_times = []
        for _ in range(10):
            start = time.perf_counter()
            output = model_copy(X)
            # Force computation to complete
            _ = output.data
            forward_times.append(time.perf_counter() - start)
        
        # Benchmark backward pass
        backward_times = []
        for _ in range(10):
            optimizer.zero_grad()
            output = model_copy(X)
            loss = loss_fn(output, y)
            start = time.perf_counter()
            loss.backward()
            # Force computation to complete
            for p in model_copy.parameters():
                if p.grad is not None:
                    _ = p.grad.data
            backward_times.append(time.perf_counter() - start)
            optimizer.step()
        
        # Benchmark full epoch
        epoch_times = []
        for _ in range(epochs):
            start = time.perf_counter()
            # Simulate one epoch
            for _ in range(10):  # 10 mini-batches
                optimizer.zero_grad()
                output = model_copy(X)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
            epoch_times.append(time.perf_counter() - start)
        
        results[backend] = {
            'forward_time': np.mean(forward_times),
            'backward_time': np.mean(backward_times),
            'epoch_time': np.mean(epoch_times)
        }
    
    return results


def compare_models(models: Dict[str, Module], x: Tensor, y: Tensor, loss_fn: Any, 
                  backends: List[str] = ['numpy']) -> Dict[str, Any]:
    """Compare multiple models."""
    results = {}
    
    for name, model in models.items():
        print(f"\nBenchmarking model: {name}")
        results[name] = benchmark_model(model, x, y, loss_fn, None, backends)
    
    # Create comparison summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for name, result in results.items():
        model_info = result['model_info']
        print(f"\n{name}:")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Memory: {model_info['parameter_memory_mb']:.2f} MB")
        
        for backend, backend_result in result['backends'].items():
            if 'error' not in backend_result:
                forward_time = backend_result['forward']['mean_time'] * 1000
                backward_time = backend_result['backward']['mean_time'] * 1000
                print(f"  {backend}: Forward {forward_time:.2f}ms, Backward {backward_time:.2f}ms")
    
    return results
