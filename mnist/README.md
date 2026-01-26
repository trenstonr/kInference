# MNIST with PyTorch

Load and use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset via PyTorch and torchvision. This project uses a **virtual environment** so dependencies stay isolated.

## Quick start

```bash
cd mnist
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python load_mnist.py
```

See [docs/SETUP.md](docs/SETUP.md) for detailed setup. See [docs/API.md](docs/API.md) for the `load_mnist` API.

## Virtual environment

A venv is already created at `mnist/venv/`. Activate it before installing or running:

| Shell    | Command |
|----------|---------|
| macOS/Linux | `source venv/bin/activate` |
| Windows (cmd) | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |

When active, your prompt usually shows `(venv)`.

To create the venv from scratch (e.g. after cloning):

```bash
cd mnist
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Download MNIST and print dataset stats:**

```bash
python load_mnist.py
```

**Use in your own code:**

```python
from load_mnist import load_mnist, DATA_DIR

train_loader = load_mnist(batch_size=64, train=True)
test_loader = load_mnist(batch_size=64, train=False)

for images, labels in train_loader:
    # images: (N, 1, 28, 28), labels: (N,)
    ...
```

Data is downloaded on first run and stored under `mnist/data/`.

## Project layout

```
mnist/
├── venv/              # Virtual environment (activate before use)
├── data/              # MNIST files (created on first run)
├── docs/              # Setup and API documentation
├── load_mnist.py      # Dataset loading API
├── requirements.txt   # PyTorch deps
└── README.md          # This file
```

## Docs

- **[docs/SETUP.md](docs/SETUP.md)** — Virtual environment and install steps
- **[docs/API.md](docs/API.md)** — `load_mnist` and helpers
