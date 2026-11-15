"""Saving and loading trained models."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import MLP


def save_checkpoint(model: MLP, path: str | Path, **metadata) -> Path:
    """Save weights + architecture so the model can be rebuilt without the training code."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": model.config, "state_dict": model.state_dict(), "metadata": metadata},
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[MLP, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    cfg = ckpt["model_config"]
    model = MLP(
        hidden=tuple(cfg["hidden"]),
        in_features=cfg["in_features"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("metadata", {})
