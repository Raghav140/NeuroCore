"""Command-line interface for NNFS."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import nnfs
from nnfs.core import set_backend, get_backend_name
from nnfs.layers import Dense, ReLU, Sigmoid
from nnfs.optim import SGD
from nnfs.utils import make_binary_classification, make_xor, train_test_split


def build_simple_model(input_size: int, hidden_size: int = 64, output_size: int = 1):
    """Build a simple neural network."""
    return nnfs.Sequential(
        Dense(input_size, hidden_size),
        ReLU(),
        Dense(hidden_size, hidden_size),
        ReLU(),
        Dense(hidden_size, output_size),
        Sigmoid()
    )


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def train_model(args):
    """Train a model using command-line arguments."""
    print_header("🚀 NNFS Training CLI")
    
    # Display configuration
    print_section("Configuration")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Backend: {get_backend_name()}")
    
    # Set backend if specified
    if args.backend:
        set_backend(args.backend)
        print(f"  Backend set to: {args.backend}")
    
    # Generate dataset
    print_section("Dataset Generation")
    if args.dataset == "xor":
        X, y = make_xor(n_samples=1000, noise=0.1, random_state=42)
        input_size = 2
        print("  Generated XOR dataset (1000 samples, noise=0.1)")
    elif args.dataset == "binary":
        X, y = make_binary_classification(n_samples=1000, n_features=2, random_state=42)
        input_size = 2
        print("  Generated binary classification dataset (1000 samples, 2 features)")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Build model
    print_section("Model Architecture")
    model = build_simple_model(input_size, args.hidden_size)
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Architecture: {input_size} → {args.hidden_size} → {args.hidden_size} → 1")
    
    # Setup optimizer and loss
    optimizer = SGD(list(model.parameters()), lr=args.lr, momentum=args.momentum)
    loss_fn = nnfs.BCELoss()
    
    # Setup trainer
    config = nnfs.TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        debug_mode=args.verbose,
        log_interval=max(1, args.epochs // 10)
    )
    
    trainer = nnfs.Trainer(model, loss_fn, optimizer, config)
    
    # Train
    print_section("Training")
    history = trainer.fit(X_train, y_train, X_test, y_test)
    
    # Evaluate
    train_loss, train_acc = trainer.evaluate(X_train, y_train)
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    
    # Final results
    print_section("Results")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Save model if requested
    if args.save_model:
        print_section("Saving Model")
        model_path = args.save_model
        trainer.save_model(model_path)
        print(f"  Model saved to: {model_path}")
    
    # Save history if requested
    if args.save_history:
        print_section("Saving History")
        history_path = args.save_history
        trainer.save_history(history_path)
        print(f"  History saved to: {history_path}")
    
    print_header("✅ Training Complete")


def benchmark_models(args):
    """Benchmark different models and backends."""
    print_header("🔬 NNFS Benchmark CLI")
    
    print_section("Configuration")
    print(f"  Backends: {args.backends}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Runs per benchmark: 20")
    
    # Test data
    print_section("Test Data")
    X, y = make_xor(n_samples=500, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  XOR dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Models to benchmark
    models = {
        'small': build_simple_model(2, 16),
        'medium': build_simple_model(2, 64),
        'large': build_simple_model(2, 128)
    }
    
    print_section("Models")
    for name, model in models.items():
        params = sum(p.data.size for p in model.parameters())
        print(f"  {name}: {params:,} parameters")
    
    # Benchmark each model
    print_section("Benchmarking")
    results = {}
    for model_name, model in models.items():
        print(f"\n  Benchmarking {model_name} model...")
        
        # Setup
        optimizer = SGD(list(model.parameters()), lr=0.01)
        loss_fn = nnfs.BCELoss()
        
        # Benchmark
        from nnfs.utils.benchmark import benchmark_backends
        model_results = {
            'backends': benchmark_backends(model, X_train, y_train, loss_fn, optimizer, args.backends.split(',')),
            'model_info': {
                'total_parameters': sum(p.data.size for p in model.parameters()),
                'trainable_parameters': sum(p.data.size for p in model.parameters() if p.requires_grad),
                'parameter_memory_bytes': sum(p.data.nbytes for p in model.parameters())
            }
        }
        
        results[model_name] = model_results
        
        # Display results
        for backend, backend_results in model_results['backends'].items():
            if isinstance(backend_results, dict) and 'forward_time' in backend_results:
                forward_time = backend_results['forward_time'] * 1000
                backward_time = backend_results['backward_time'] * 1000
                print(f"    {backend}: Forward {forward_time:.2f}ms, Backward {backward_time:.2f}ms")
    
    # Save results
    if args.output:
        print_section("Saving Results")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to: {args.output}")
    
    print_header("✅ Benchmark Complete")


def show_info(args):
    """Show system information."""
    print_header("ℹ️  NNFS System Information")
    
    print_section("Framework")
    print(f"  Version: {nnfs.__version__}")
    print(f"  Author: {nnfs.__author__}")
    
    print_section("Backend")
    print(f"  Current: {get_backend_name()}")
    print(f"  Available: numpy" + (", cupy" if nnfs.core.backend.CUPY_AVAILABLE else ""))
    
    print_section("Installation")
    print(f"  Python: {sys.version}")
    print(f"  Package location: {nnfs.__file__}")
    
    print_header("✅ Info Complete")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NNFS - Neural Network From Scratch CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nnfs-train --dataset xor --epochs 100 --lr 0.1
  nnfs-train --dataset binary --epochs 50 --hidden-size 128
  nnfs-benchmark --backends numpy --epochs 20
  nnfs-info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser(
        'train', 
        help='Train a neural network',
        description='Train a neural network on various datasets'
    )
    train_parser.add_argument('--dataset', choices=['xor', 'binary'], default='xor',
                             help='Dataset to use for training')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=0.01,
                             help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--hidden-size', type=int, default=64,
                             help='Hidden layer size')
    train_parser.add_argument('--momentum', type=float, default=0.0,
                             help='SGD momentum')
    train_parser.add_argument('--backend', choices=['numpy', 'cupy', 'auto'],
                             help='Backend to use')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Enable verbose logging')
    train_parser.add_argument('--save-model', type=str,
                             help='Path to save trained model')
    train_parser.add_argument('--save-history', type=str,
                             help='Path to save training history')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark', 
        help='Benchmark models',
        description='Benchmark model performance across different backends'
    )
    benchmark_parser.add_argument('--backends', type=str, default='numpy',
                                 help='Comma-separated list of backends to test')
    benchmark_parser.add_argument('--epochs', type=int, default=20,
                                 help='Number of epochs for benchmarking')
    benchmark_parser.add_argument('--output', type=str,
                                 help='Path to save benchmark results')
    
    # Info command
    info_parser = subparsers.add_parser(
        'info', 
        help='Show system information',
        description='Display framework and system information'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'benchmark':
        benchmark_models(args)
    elif args.command == 'info':
        show_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
