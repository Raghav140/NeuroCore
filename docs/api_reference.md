# API Reference (Core)

Project: **NeuroCore** (Python import name: `ncfs`)

## `ncfs.core.tensor`

- `Tensor(data, requires_grad=False)`
- `tensor(data, requires_grad=False)`
- Autograd-enabled ops: `+`, `-`, `*`, `/`, `@`, `sum`, `mean`, `exp`, `log`, `tanh`, `relu`, `reshape`, `backward`

## `ncfs.core.module`

- `Module`
  - `forward`, `__call__`, `parameters`, `named_parameters`, `zero_grad`
  - `train`, `eval`
  - hook registration
- `Parameter`

## `ncfs.layers`

- `Dense`
- `BatchNorm1d`
- `Dropout`
- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Embedding`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`

## `ncfs.optim`

- `SGD`
- `StepLR`
- `ReduceLROnPlateau`

## `ncfs.core.trainer`

- `Trainer`
- `TrainerConfig`
- `Callback`

## `ncfs.utils`

- datasets: `make_xor`, `make_binary_classification`, `make_regression`
- metrics: `accuracy_score`, `f1_binary`, `confusion_matrix_binary`
- summary: `model_summary`
- benchmark: `benchmark_backends`, `print_benchmark_table`, `save_benchmark_json`
