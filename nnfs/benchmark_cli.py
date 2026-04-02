"""CLI for backend benchmarking."""

from __future__ import annotations

import argparse

from .utils import benchmark_backends, print_benchmark_table, save_benchmark_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NNFS backend benchmarks.")
    parser.add_argument(
        "--backends",
        type=str,
        default="numpy,cupy",
        help="Comma-separated backends to benchmark (e.g. numpy,cupy).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Benchmark training epochs.")
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = benchmark_backends(backends=backends, epochs=args.epochs)
    print_benchmark_table(results)
    save_benchmark_json(results, args.out)
    print(f"\nSaved benchmark report: {args.out}")


if __name__ == "__main__":
    main()
