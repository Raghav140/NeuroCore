## NeuroCore: Library-Level Deep Learning Framework

This repository now includes a professionalized deep learning library named **NeuroCore** (package import: `nnfs`) with a PyTorch-inspired API, optional GPU backend switching (NumPy/CuPy), a reusable trainer, Streamlit dashboard, packaging metadata, CLI, and tests.

The original educational code in `nn_from_scratch` is preserved for backward compatibility.

## What Is Included

- `nnfs/core`
  - `Module`, `Parameter` with automatic registration, hooks, nested parameter traversal
  - `Sequential` container
  - `Trainer` + callback system + JSON history export
  - backend abstraction (`numpy`/`cupy`) in `backend.py`
  - losses: `MSELoss`, `BCELoss`, `CrossEntropyLoss`
- `nnfs/layers`
  - `Dense`, `BatchNorm1d`, `Dropout`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `nnfs/optim`
  - `SGD` (momentum + gradient clipping)
  - `StepLR`, `ReduceLROnPlateau`
- `nnfs/utils`
  - synthetic datasets, metrics, model summary
- `app.py`
  - Streamlit dashboard for interactive training and visualization
- `nnfs/cli.py`
  - CLI entrypoint for command-line training (`nnfs-train`)
- `pyproject.toml`
  - pip-installable package metadata and script entrypoint
- `requirements.txt`
  - dashboard/runtime dependencies

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dashboard]"
python -m pip install -e ".[gpu]"
```

## Quick Usage

```python
import numpy as np
from nnfs import Sequential, Dense, ReLU, Sigmoid, BCELoss, Trainer
from nnfs.optim import SGD
from nnfs.core.trainer import TrainerConfig
from nnfs.utils import make_xor

np.random.seed(42)
X, y = make_xor(400)
model = Sequential(Dense(2, 16, init="he"), ReLU(), Dense(16, 1), Sigmoid())
trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.1))
history = trainer.fit(X, y, TrainerConfig(epochs=800, batch_size=32))
```

## Backend Selection (NumPy/CuPy)

```python
from nnfs import set_backend, get_backend_name

set_backend("auto")   # picks CuPy when available, else NumPy
print(get_backend_name())
```

If CuPy is not installed, backend safely falls back to NumPy.

## Streamlit Dashboard

Run:

```bash
streamlit run app.py
```

Features:

- dataset choice (XOR / synthetic binary classification)
- architecture and hyperparameter controls
- live loss and accuracy curves
- decision boundary for 2D datasets
- accuracy, F1, and confusion matrix display

## CLI

After editable install:

```bash
nnfs-train --epochs 800 --lr 0.1 --batch-size 32
```

Benchmark CLI:

```bash
nnfs-benchmark --backends numpy,cupy --epochs 20 --out benchmark.json
```

## Testing

```bash
python -m unittest discover -s tests -v
```

Current tests cover:

- Module system and parameter registration
- Sequential training behavior
- backend consistency checks
- dashboard helper logic
- package import and legacy framework tests

## Architecture Diagram

```text
                +----------------------+
                |      Trainer         |
                | fit/eval/logging     |
                +----------+-----------+
                           |
                 uses      v
+------------------+   +----------------------+   +------------------+
|   nnfs.optim     |   |     nnfs.core        |   |   nnfs.layers    |
| SGD/Schedulers   |-->| Module/Tensor/Loss   |<--| Dense/Conv/Pool  |
+------------------+   +----------------------+   +------------------+
                           |
                           v
                  +-------------------+
                  | backend.py (xp)   |
                  | NumPy / CuPy      |
                  +-------------------+
```

## How Autograd Works Internally

`Tensor` in `nnfs/core/tensor.py` builds a dynamic graph during forward pass:

1. Every tensor op (add/mul/matmul/exp/log/relu/...) creates a new `Tensor`.
2. The new tensor stores:
   - parents (`_children`)
   - operation metadata (`_op`)
   - a local `_backward()` closure.
3. Calling `loss.backward()`:
   - topologically sorts the graph from loss to leaves,
   - runs local `_backward()` in reverse order,
   - accumulates gradients into leaf parameters.

This removes most manual layer-level derivative code and enables PyTorch-style training flow.

## GPU Backend (NumPy/CuPy)

Backend selection is in `nnfs/core/backend.py`:

- `set_backend("numpy")` for CPU
- `set_backend("cupy")` for GPU (if installed)
- `set_backend("auto")` for automatic selection

All tensor storage and math route through backend-aware array ops. If CuPy is unavailable, it safely falls back to NumPy.

## Advanced Usage

### CNN Example

```python
from nnfs import Sequential, Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Sigmoid, BCELoss, Trainer
from nnfs.optim import SGD
from nnfs.core.trainer import TrainerConfig

model = Sequential(
    Conv2D(1, 4, kernel_size=3, padding=1),
    ReLU(),
    MaxPooling2D(2),
    Flatten(),
    Dense(4 * 4 * 4, 16),
    ReLU(),
    Dense(16, 1),
    Sigmoid(),
)
trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.01))
```

### Debug Mode and Safety Checks

`TrainerConfig(debug_mode=True)` enables:

- NaN/Inf gradient detection
- gradient explosion detection via `gradient_explosion_threshold`

## Benchmarking

Use the utility module:

```python
from nnfs.utils import benchmark_backends, print_benchmark_table, save_benchmark_json

results = benchmark_backends(["numpy", "cupy"], epochs=20)
print_benchmark_table(results)
save_benchmark_json(results, "benchmark.json")
```

Outputs include forward time, backward time, epoch time, and throughput.

You can also run benchmark end-to-end from CLI:

```bash
nnfs-benchmark --backends numpy,cupy --epochs 20 --out benchmark.json
```

This prints a console table and saves a JSON report in one command.

## Model Zoo

Scripts under `examples/`:

- `xor.py`
- `binary_classification.py`
- `mnist_mlp.py`
- `cnn_example.py`

Run:

```bash
python examples/xor.py
python examples/binary_classification.py
python examples/mnist_mlp.py
python examples/cnn_example.py
```

## Docs Folder

- `docs/tutorials.md`
- `docs/api_reference.md`
- `docs/autograd_internals.md`

These provide guided workflows and API-level references for both beginner and advanced users.

## Creator

- **Raghav Sharma**

## Complete Project Structure

At repository level, this project is organized as:

- `nnfs/` - library implementation (core, layers, optim, utils, CLI)
- `nn_from_scratch/` - original educational framework (kept for backward compatibility)
- `examples/` - runnable model zoo scripts
- `docs/` - tutorials and API/autograd documentation
- `tests/` - automated test suite
- `app.py` - Streamlit dashboard entrypoint
- `pyproject.toml` - package metadata and CLI registration
- `requirements.txt` - runtime dependencies for dashboard usage

## Run Everything (Recommended Validation Order)

Use this order to validate the full project on a fresh machine:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[dashboard]"
python -m unittest discover -s tests -v
python -m nn_from_scratch.main
nnfs-train --epochs 200 --lr 0.1 --batch-size 64
nnfs-benchmark --backends numpy,cupy --epochs 20 --out benchmark.json
python -m streamlit run app.py
```

## Notes on Example Outputs

- `examples/xor.py`: should typically reach very high accuracy (often near `1.0`).
- `examples/binary_classification.py`: expected to reach strong accuracy (commonly `0.85+`).
- `examples/mnist_mlp.py`: currently uses a portable random-data scaffold (no external MNIST download dependency), so treat it as architecture/training-pipeline validation.
- `examples/cnn_example.py`: validates CNN forward/backward/autograd path and debug-mode compatibility.

## Packaging and Distribution Notes

- Project name: **NeuroCore**
- Package name: `neurocore-numpy`
- Installed import: `import nnfs`
- CLI commands exposed via `pyproject.toml`:
  - `nnfs-train`
  - `nnfs-benchmark`

If the shell cannot resolve a CLI command immediately after install on Windows, use:

```bash
.\.venv\Scripts\nnfs-train.exe
.\.venv\Scripts\nnfs-benchmark.exe
```

## Troubleshooting

- If `streamlit` command is not found, run it with Python module mode:
  - `python -m streamlit run app.py`
- If `cupy` is unavailable, backend automatically falls back to NumPy.
- If debug mode raises gradient explosion/NaN errors, lower learning rate or enable gradient clipping in optimizer settings.

