"""Configurable multilayer perceptron."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected classifier: ``in_features -> hidden... -> num_classes``.

    Args:
        hidden: Sizes of the hidden layers, e.g. ``(256, 128)``. Empty = logistic regression.
        dropout: Dropout probability after each hidden activation (0 disables it).
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (128,),
        in_features: int = 28 * 28,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {
            "hidden": list(hidden),
            "in_features": in_features,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
