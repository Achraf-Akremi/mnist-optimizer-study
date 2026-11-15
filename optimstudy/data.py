"""MNIST loading into memory with a fixed train/validation/test split.

The whole dataset (~200 MB as float32) is kept as tensors, which makes small MLPs
train many times faster than a per-sample ``Dataset``/``DataLoader`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class Split:
    x: torch.Tensor  # (N, 784) float32, normalized
    y: torch.Tensor  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)

    def to(self, device: str | torch.device) -> "Split":
        return Split(self.x.to(device), self.y.to(device))


@dataclass
class Data:
    train: Split
    val: Split
    test: Split

    def to(self, device: str | torch.device) -> "Data":
        return Data(self.train.to(device), self.val.to(device), self.test.to(device))


def _prepare(images: torch.Tensor, labels: torch.Tensor) -> Split:
    x = images.float().div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
    return Split(x.reshape(len(x), -1), labels.long().clone())


def load_mnist(
    root: str = "data",
    val_size: int = 10_000,
    split_seed: int = 0,
    limit_train: int | None = None,
) -> Data:
    """Download MNIST if needed and return normalized train/val/test splits.

    The validation set is carved out of the official 60k training images, so the
    10k test images are only used for the final evaluation.
    """
    from torchvision import datasets  # imported lazily so tests don't need it

    train_full = datasets.MNIST(root=root, train=True, download=True)
    test = datasets.MNIST(root=root, train=False, download=True)

    full = _prepare(train_full.data, train_full.targets)
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(split_seed))
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    if limit_train is not None:
        train_idx = train_idx[:limit_train]

    return Data(
        train=Split(full.x[train_idx], full.y[train_idx]),
        val=Split(full.x[val_idx], full.y[val_idx]),
        test=_prepare(test.data, test.targets),
    )


def batches(
    split: Split,
    batch_size: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x, y)`` mini-batches from an in-memory split."""
    n = len(split)
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(split.x.device)
    else:
        idx = torch.arange(n, device=split.x.device)
    for start in range(0, n, batch_size):
        b = idx[start : start + batch_size]
        yield split.x[b], split.y[b]
