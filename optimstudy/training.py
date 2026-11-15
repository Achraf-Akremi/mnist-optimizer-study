"""Training and evaluation of a single run."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .data import Data, Split, batches
from .model import MLP
from .optimizers import make_optimizer


@dataclass
class RunConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    hidden: tuple[int, ...] = (128,)
    epochs: int = 3
    batch_size: int = 128
    dropout: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = "-".join(map(str, self.hidden)) or "none"
        return d


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)  # mean loss per epoch
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    diverged: bool = False
    seconds: float = 0.0


@torch.no_grad()
def evaluate(model: nn.Module, split: Split, batch_size: int = 2048) -> tuple[float, float]:
    """Return ``(mean cross-entropy loss, accuracy)`` on a split."""
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in batches(split, batch_size):
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
    n = len(split)
    return total_loss / n, correct / n


@torch.no_grad()
def confusion_matrix(model: nn.Module, split: Split, num_classes: int = 10) -> torch.Tensor:
    """Rows = true class, columns = predicted class."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in batches(split, 2048):
        pred = model(x).argmax(1)
        idx = (y * num_classes + pred).cpu()
        cm += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm


def fit(cfg: RunConfig, data: Data, verbose: bool = False) -> tuple[MLP, History]:
    """Train one model according to ``cfg``. Data must already be on the target device."""
    device = data.train.x.device
    torch.manual_seed(cfg.seed)
    model = MLP(cfg.hidden, dropout=cfg.dropout).to(device)
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    shuffle_gen = torch.Generator().manual_seed(cfg.seed)

    history = History()
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        running, steps = 0.0, 0
        for x, y in batches(data.train, cfg.batch_size, shuffle=True, generator=shuffle_gen):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        val_loss, val_acc = evaluate(model, data.val)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        if verbose:
            print(f"  epoch {epoch + 1}/{cfg.epochs}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  val acc {val_acc:.4f}")

        if not math.isfinite(train_loss):  # learning rate too high, stop wasting time
            history.diverged = True
            break

    history.seconds = time.perf_counter() - start
    return model, history
