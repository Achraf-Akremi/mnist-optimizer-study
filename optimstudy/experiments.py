"""Running grids of configurations and aggregating results over seeds."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

from .data import Data
from .model import MLP
from .training import RunConfig, fit


def build_grid(
    base: RunConfig,
    optimizers: list[str],
    lrs: list[float],
    hiddens: list[tuple[int, ...]],
    seeds: list[int],
) -> list[RunConfig]:
    return [
        replace(base, optimizer=o, lr=lr, hidden=h, seed=s)
        for o, lr, h, s in product(optimizers, lrs, hiddens, seeds)
    ]


def run_grid(configs: list[RunConfig], data: Data, out_dir: str | Path) -> list[dict]:
    """Train every config, writing ``runs.csv`` and ``histories.json`` as it goes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for i, cfg in enumerate(configs, start=1):
        model, hist = fit(cfg, data)
        row = {
            **cfg.as_dict(),
            "params": MLP(cfg.hidden, dropout=cfg.dropout).num_parameters(),
            "final_train_loss": hist.train_loss[-1],
            "final_val_loss": hist.val_loss[-1],
            "final_val_acc": hist.val_acc[-1],
            "diverged": hist.diverged,
            "seconds": round(hist.seconds, 2),
            "history": {"train_loss": hist.train_loss, "val_loss": hist.val_loss, "val_acc": hist.val_acc},
        }
        rows.append(row)
        flag = "  (diverged)" if hist.diverged else ""
        print(f"[{i:3d}/{len(configs)}] {cfg.optimizer:>8} lr={cfg.lr:<8g} hidden={row['hidden']:<12} "
              f"seed={cfg.seed}  val acc {row['final_val_acc']:.4f}{flag}")

        _write_rows(rows, out_dir)

    return rows


def _write_rows(rows: list[dict], out_dir: Path) -> None:
    scalar_keys = [k for k in rows[0] if k != "history"]
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in scalar_keys} for r in rows)
    with open(out_dir / "histories.json", "w") as f:
        json.dump(rows, f, indent=1)


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    return statistics.fmean(finite), (statistics.stdev(finite) if len(finite) > 1 else 0.0)


def aggregate(rows: list[dict], keys: tuple[str, ...] = ("optimizer", "lr", "hidden")) -> list[dict]:
    """Average final validation metrics over seeds for each unique combination of ``keys``."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)

    summary = []
    for key, group in groups.items():
        acc_mean, acc_std = _mean_std([g["final_val_acc"] for g in group])
        loss_mean, loss_std = _mean_std([g["final_val_loss"] for g in group])
        summary.append({
            **dict(zip(keys, key)),
            "params": group[0]["params"],
            "seeds": len(group),
            "val_acc_mean": acc_mean,
            "val_acc_std": acc_std,
            "val_loss_mean": loss_mean,
            "val_loss_std": loss_std,
            "diverged_runs": sum(g["diverged"] for g in group),
        })
    return summary


def best_per_optimizer(summary: list[dict]) -> dict[str, dict]:
    """Pick the configuration with the highest mean validation accuracy for each optimizer."""
    best: dict[str, dict] = {}
    for s in summary:
        if math.isnan(s["val_acc_mean"]):
            continue
        cur = best.get(s["optimizer"])
        if cur is None or s["val_acc_mean"] > cur["val_acc_mean"]:
            best[s["optimizer"]] = s
    return best


def write_csv(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
