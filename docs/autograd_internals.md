# Autograd Internals

This document explains how **NeuroCore** (`ncfs`) automatic differentiation works step by step.

## Core Tensor Fields

Each `Tensor` stores:

- `data`: backend array (`numpy.ndarray` or `cupy.ndarray`)
- `grad`: accumulated gradient (same shape as `data`, when `requires_grad=True`)
- `requires_grad`: whether to track operations for backward
- `_children`: parent tensors used to produce this tensor
- `_op`: operation label (e.g. `"+"`, `"matmul"`, `"relu"`)
- `_backward`: local function that propagates gradient to parents

## Dynamic Graph Construction

Graph is built during forward pass. Example:

```python
x = Tensor([[2.0]], requires_grad=True)
y = x * x + x
z = y.sum()
```

Computation graph:

```text
 x ----*----\
  \         +---- y ---- sum ---- z
   \----x---/
```

- `x * x` creates node `*` with parents `(x, x)`
- `(...)+x` creates node `+` with parents `(*, x)`
- `sum()` creates node `sum` with parent `(+)`

## Backward Pass Algorithm

`z.backward()` performs:

1. **Seed gradient**
   - if scalar output: initialize `z.grad = 1`
2. **Topological sort**
   - DFS through `_children`
   - collect nodes in dependency order
3. **Reverse traversal**
   - run each node’s `_backward()` from output to leaves
   - accumulate into parent `grad` buffers

Pseudo-flow:

```text
build_topo(z)
for node in reversed(topo):
    node._backward()
```

## Chain Rule in Local Backward

Each op defines local derivative logic. Example for multiplication:

```text
out = a * b
dL/da += dL/dout * b
dL/db += dL/dout * a
```

Matrix multiplication:

```text
out = A @ B
dL/dA += dL/dout @ B^T
dL/dB += A^T @ dL/dout
```

Broadcast-aware ops call an internal unbroadcast helper so gradients are reduced to source shapes.

## Integration with Modules

- `Parameter` extends `Tensor(requires_grad=True)`
- Layers operate on `Tensor` objects
- Loss returns scalar `Tensor`
- Trainer calls `loss.backward()` then optimizer updates `Parameter.data`

## Debug Tips

- Enable `TrainerConfig(debug_mode=True)` for:
  - NaN/Inf gradient detection
  - gradient explosion threshold checks
- Use small batches and deterministic seeds when debugging custom layers.
