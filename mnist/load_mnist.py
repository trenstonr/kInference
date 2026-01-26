"""
MNIST dataset setup and loading using PyTorch.
Downloads the dataset on first run and provides train/test loaders.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Default data directory (downloads here on first run)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_mnist_transforms(train: bool = True):
    """Standard transforms for MNIST (normalize to mean 0.1307, std 0.3081)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_mnist(
    root: str | Path = DATA_DIR,
    batch_size: int = 64,
    train: bool = True,
    download: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Load MNIST as PyTorch DataLoaders.

    Args:
        root: Directory for dataset. Default: mnist/data/
        batch_size: Batch size for the DataLoader.
        train: If True, load training set; else test set.
        download: If True, download MNIST if not present.
        num_workers: DataLoader workers (0 = main process only).
        shuffle: Shuffle batches (typically True for train, False for test).

    Returns:
        DataLoader over MNIST images and labels.
    """
    root = Path(root)
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=get_mnist_transforms(train=train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    """Download MNIST and print basic stats."""
    print("Setting up MNIST...")
    train_loader = load_mnist(batch_size=64, train=True, shuffle=True)
    test_loader = load_mnist(batch_size=64, train=False, shuffle=False)

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    print(f"Train samples: {train_size}")
    print(f"Test samples:  {test_size}")

    # Optional: show one batch shape
    images, labels = next(iter(train_loader))
    print(f"Batch shape: images {images.shape}, labels {labels.shape}")
    print("MNIST is ready.")


if __name__ == "__main__":
    main()
