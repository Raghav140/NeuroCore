"""Benchmark utilities for CPU/GPU backends."""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np

from .. import BCELoss, Dense, ReLU, Sequential, Sigmoid, Trainer, set_backend
from ..core.trainer import TrainerConfig
from ..optim import SGD
from .datasets import make_xor


def benchmark_backends(backends: List[str] = ("numpy", "cupy"), epochs: int = 20) -> List[Dict]:
    """Benchmark forward/backward and epoch throughput across backends."""
    results = []
    for backend in backends:
        try:
            set_backend(backend)
        except Exception:
            continue
        np.random.seed(42)
        X, y = make_xor(512)
        model = Sequential(Dense(2, 32, init="he"), ReLU(), Dense(32, 1), Sigmoid())
        trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.05))

        t0 = time.perf_counter()
        preds = model(X)
        fwd = time.perf_counter() - t0

        t1 = time.perf_counter()
        loss = BCELoss()(preds, y)
        loss.backward()
        bwd = time.perf_counter() - t1

        t2 = time.perf_counter()
        trainer.fit(X, y, TrainerConfig(epochs=epochs, batch_size=64))
        epoch_total = time.perf_counter() - t2

        results.append(
            {
                "backend": backend,
                "forward_ms": 1000 * fwd,
                "backward_ms": 1000 * bwd,
                "epoch_ms": 1000 * epoch_total / epochs,
                "samples_per_sec": (X.shape[0] * epochs) / max(epoch_total, 1e-12),
            }
        )
    return results


def print_benchmark_table(results: List[Dict]) -> None:
    """Print benchmarking results in a simple console table."""
    if not results:
        print("No benchmark results available.")
        return
    header = f"{'backend':10s} {'forward_ms':>12s} {'backward_ms':>12s} {'epoch_ms':>12s} {'samples/s':>12s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['backend']:10s} {r['forward_ms']:12.3f} {r['backward_ms']:12.3f} "
            f"{r['epoch_ms']:12.3f} {r['samples_per_sec']:12.2f}"
        )


def save_benchmark_json(results: List[Dict], path: str) -> None:
    """Save benchmark results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
