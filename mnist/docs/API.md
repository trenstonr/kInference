# load_mnist API reference

This document describes the public API of `load_mnist.py`.

## Module-level constant

### `DATA_DIR`

- **Type:** `pathlib.Path`
- **Meaning:** Default directory where MNIST is stored. Resolves to `mnist/data/` relative to the script. The directory is created if it doesn’t exist.

**Example:**

```python
from load_mnist import DATA_DIR
print(DATA_DIR)  # e.g. /path/to/mnist/data
```

---

## Functions

### `get_mnist_transforms(train=True)`

Returns the standard transform pipeline used for MNIST in this project.

- **Parameters:**
  - `train` (`bool`, optional): Ignored by the current implementation; kept for a possible train/test split of augmentations. Default `True`.
- **Returns:** `torchvision.transforms.Compose` with:
  - `ToTensor()` — PIL/image → tensor, shape `(1, 28, 28)`, values in `[0, 1]`
  - `Normalize((0.1307,), (0.3081,))` — standard MNIST mean and std

**Example:**

```python
from load_mnist import get_mnist_transforms
from torchvision import datasets

t = get_mnist_transforms()
ds = datasets.MNIST(root="./data", train=True, download=True, transform=t)
```

---

### `load_mnist(root=DATA_DIR, batch_size=64, train=True, download=True, num_workers=0, shuffle=True)`

Builds a `DataLoader` for the MNIST train or test set.

- **Parameters:**
  - **`root`** (`str` or `pathlib.Path`, optional): Root directory for the dataset. Default is `DATA_DIR` (`mnist/data/`).
  - **`batch_size`** (`int`, optional): Batch size. Default `64`.
  - **`train`** (`bool`, optional): `True` → training set (60,000 samples), `False` → test set (10,000 samples). Default `True`.
  - **`download`** (`bool`, optional): If `True`, download MNIST when not present. Default `True`.
  - **`num_workers`** (`int`, optional): Number of DataLoader workers. `0` = main process only. Default `0`.
  - **`shuffle`** (`bool`, optional): Shuffle batches. Typically `True` for training, `False` for testing. Default `True`.

- **Returns:** `torch.utils.data.DataLoader`  
  - Each batch is `(images, labels)`:
    - `images`: `(N, 1, 28, 28)`, dtype `float32`, normalized
    - `labels`: `(N,)`, dtype `int64`, values in `0..9`  
  - `pin_memory` is set to `True` when CUDA is available.

**Examples:**

```python
from load_mnist import load_mnist

# Default: train loader, batch 64, shuffled
train_loader = load_mnist()

# Test set, no shuffle
test_loader = load_mnist(train=False, shuffle=False)

# Custom batch size and data path
loader = load_mnist(root="/tmp/mnist", batch_size=128, train=True)
```

---

### `main()`

Entry point when the script is run with `python load_mnist.py`. It:

1. Builds train and test loaders with default args.
2. Prints train/test sample counts.
3. Prints the shape of one batch.
4. Does not accept arguments.

---

## Data format

- **Images:** `(N, 1, 28, 28)` — N images, 1 channel, 28×28 pixels, normalized with mean `0.1307`, std `0.3081`.
- **Labels:** `(N,)` — class indices in `{0, 1, …, 9}`.

Example loop:

```python
for images, labels in load_mnist(batch_size=32, train=True):
    # images: (32, 1, 28, 28)
    # labels: (32,)
    break
```
