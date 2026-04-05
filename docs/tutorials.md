# Tutorials

Project: **NeuroCore** (Python import name: `ncfs`)

## 1) Quick Start (XOR)

```python
import numpy as np
from ncfs import Sequential, Dense, ReLU, Sigmoid, BCELoss, Trainer
from ncfs.optim import SGD
from ncfs.core.trainer import TrainerConfig
from ncfs.utils import make_xor

np.random.seed(42)
X, y = make_xor(400)
model = Sequential(Dense(2, 16, init="he"), ReLU(), Dense(16, 1), Sigmoid())
trainer = Trainer(model, BCELoss(), SGD(model.parameters(), lr=0.1))
trainer.fit(X, y, TrainerConfig(epochs=700, batch_size=64))
```

## 2) Enable GPU backend (if CuPy installed)

```python
from ncfs import set_backend, get_backend_name
set_backend("auto")
print(get_backend_name())
```

## 3) Run interactive dashboard

```bash
streamlit run app.py
```

## 4) Run model zoo examples

```bash
python examples/xor.py
python examples/binary_classification.py
python examples/mnist_mlp.py
python examples/cnn_example.py
```
