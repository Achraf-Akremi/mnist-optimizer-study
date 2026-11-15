"""Optimizer factory so experiments can refer to optimizers by name."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

Factory = Callable[[Iterable[nn.Parameter], float], torch.optim.Optimizer]

OPTIMIZERS: dict[str, Factory] = {
    "sgd": lambda p, lr: torch.optim.SGD(p, lr=lr),
    "momentum": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "nesterov": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True),
    "rmsprop": lambda p, lr: torch.optim.RMSprop(p, lr=lr),
    "adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
    "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=1e-2),
}


def make_optimizer(name: str, params: Iterable[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    try:
        return OPTIMIZERS[name](params, lr)
    except KeyError:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {', '.join(OPTIMIZERS)}") from None
