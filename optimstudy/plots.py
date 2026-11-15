"""Static matplotlib figures for the study."""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .optimizers import OPTIMIZERS

# Fixed categorical order: a color always belongs to the same optimizer,
# no matter which subset of optimizers a figure shows.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OPTIMIZER_COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(OPTIMIZERS)}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OPTIMIZER_MARKERS = {name: _MARKERS[i % len(_MARKERS)] for i, name in enumerate(OPTIMIZERS)}

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _legend_outside(ax) -> None:
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)


def _new_fig(size=(10, 5.5)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    return fig, ax


def plot_lr_sweep(summary: list[dict], title: str = "Validation accuracy vs. learning rate"):
    """Mean final validation accuracy per optimizer across learning rates (band = ±1 std over seeds)."""
    fig, ax = _new_fig()
    by_opt: dict[str, list[dict]] = defaultdict(list)
    for s in summary:
        by_opt[s["optimizer"]].append(s)

    for name in [o for o in OPTIMIZERS if o in by_opt]:
        pts = sorted((s for s in by_opt[name] if not math.isnan(s["val_acc_mean"])), key=lambda s: s["lr"])
        if not pts:
            continue
        lr = np.array([p["lr"] for p in pts])
        mean = np.array([p["val_acc_mean"] for p in pts])
        std = np.array([p["val_acc_std"] for p in pts])
        color = OPTIMIZER_COLORS[name]
        ax.fill_between(lr, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(lr, mean, color=color, linewidth=2, marker=OPTIMIZER_MARKERS[name], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)

    ax.set_xscale("log")
    ax.set_ylim(top=1.0)
    _style(ax, title, "Learning rate (log scale)", "Final validation accuracy")
    _legend_outside(ax)
    fig.tight_layout()
    return fig


def plot_learning_curves(rows: list[dict], best: dict[str, dict], metric: str = "val_loss"):
    """Per-epoch curves for each optimizer at its best learning rate (mean ± std over seeds)."""
    fig, ax = _new_fig()
    for name in [o for o in OPTIMIZERS if o in best]:
        b = best[name]
        runs = [r for r in rows if r["optimizer"] == name and r["lr"] == b["lr"]
                and r["hidden"] == b["hidden"] and not r["diverged"]]
        if not runs:
            continue
        length = min(len(r["history"][metric]) for r in runs)
        curves = np.array([r["history"][metric][:length] for r in runs])
        epochs = np.arange(1, length + 1)
        mean, std = curves.mean(0), curves.std(0)
        color = OPTIMIZER_COLORS[name]
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, color=color, linewidth=2, marker=OPTIMIZER_MARKERS[name], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name} (lr={b['lr']:g})")

    nice = {"val_loss": "Validation loss", "val_acc": "Validation accuracy", "train_loss": "Training loss"}
    ax.xaxis.get_major_locator().set_params(integer=True)
    _style(ax, f"{nice.get(metric, metric)} per epoch at each optimizer's best learning rate",
           "Epoch", nice.get(metric, metric))
    _legend_outside(ax)
    fig.tight_layout()
    return fig


def plot_architecture_study(summary: list[dict]):
    """Grouped bars: one group per architecture (ordered by size), one bar per optimizer."""
    archs = sorted({(s["params"], s["hidden"]) for s in summary})
    opts = [o for o in OPTIMIZERS if any(s["optimizer"] == o for s in summary)]
    lookup = {(s["optimizer"], s["hidden"]): s for s in summary}

    fig, ax = _new_fig()
    x = np.arange(len(archs))
    width = 0.8 / len(opts)
    for i, name in enumerate(opts):
        means, stds = [], []
        for _, hidden in archs:
            row = lookup.get((name, hidden))
            means.append(row["val_acc_mean"] if row else np.nan)
            stds.append(row["val_acc_std"] if row else 0.0)
        offset = (i - (len(opts) - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.92, yerr=stds, color=OPTIMIZER_COLORS[name],
               ecolor=MUTED, capsize=3, label=name, zorder=2)

    labels = [f"{'no hidden layer' if h == 'none' else h.replace('-', ' → ')}\n{p:,} params"
              for p, h in archs]
    ax.set_xticks(x, labels)
    finite = [s["val_acc_mean"] - s["val_acc_std"] for s in summary if not math.isnan(s["val_acc_mean"])]
    if finite:
        ax.set_ylim(max(0.0, min(finite) - 0.02), 1.0)
    _style(ax, "Validation accuracy by hidden-layer architecture", "Hidden layers (smallest to largest)",
           "Final validation accuracy")
    ax.grid(False, axis="x")
    _legend_outside(ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, title: str = "Test-set confusion matrix"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    im = ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    n = len(cm)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if norm[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted digit", color=MUTED)
    ax.set_ylabel("True digit", color=MUTED)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, pad=12)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("Share of true class", color=MUTED)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig
