"""Command-line interface: ``python -m optimstudy <command> [options]``.

Commands
--------
sweep     Learning-rate sweep for several optimizers, averaged over seeds.
arch      Compare network sizes/depths for one or more optimizers.
train     Train a single model, report test metrics and save a checkpoint.
evaluate  Load a checkpoint and evaluate it on the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch

from . import plots
from .checkpoint import load_checkpoint, save_checkpoint
from .data import Data, load_mnist
from .experiments import aggregate, best_per_optimizer, build_grid, run_grid, write_csv
from .optimizers import OPTIMIZERS
from .training import RunConfig, confusion_matrix, evaluate, fit

DEFAULT_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
DEFAULT_ARCHS = ["64", "256", "128,128", "256,128", "512,256,128"]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load(args) -> Data:
    data = load_mnist(root=args.data_root, val_size=args.val_size, limit_train=args.limit_train)
    device = _device(args.device)
    print(f"Data: {len(data.train)} train / {len(data.val)} val / {len(data.test)} test on {device}")
    return data.to(device)


def _parse_arch(text: str) -> tuple[int, ...]:
    text = text.strip()
    return tuple(int(t) for t in text.split(",")) if text and text != "none" else ()


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150)
    matplotlib.pyplot.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------------- commands

def cmd_sweep(args) -> None:
    data = _load(args)
    out = Path(args.out)
    base = RunConfig(epochs=args.epochs, batch_size=args.batch_size)
    configs = build_grid(base, args.optimizers, args.lrs, [tuple(args.hidden)],
                         list(range(args.seeds)))
    print(f"Running {len(configs)} trainings...")
    rows = run_grid(configs, data, out)

    summary = aggregate(rows)
    write_csv(summary, out / "summary.csv")
    best = best_per_optimizer(summary)
    (out / "best.json").write_text(json.dumps(best, indent=2))

    print("\nBest learning rate per optimizer (mean ± std of final validation accuracy):")
    for name, b in sorted(best.items(), key=lambda kv: -kv[1]["val_acc_mean"]):
        print(f"  {name:>8}: lr={b['lr']:<8g} {b['val_acc_mean']:.4f} ± {b['val_acc_std']:.4f}")

    _save(plots.plot_lr_sweep(summary), out / "lr_sweep.png")
    _save(plots.plot_learning_curves(rows, best, "val_loss"), out / "val_loss_curves.png")


def cmd_arch(args) -> None:
    lrs: dict[str, float] = {}
    if args.best:
        best = json.loads(Path(args.best).read_text())
        lrs = {name: b["lr"] for name, b in best.items() if name in args.optimizers}
    missing = [o for o in args.optimizers if o not in lrs]
    if missing and args.lr is None:
        raise SystemExit(f"No learning rate for {missing}: pass --lr or --best results/sweep/best.json")
    for o in missing:
        lrs[o] = args.lr

    data = _load(args)
    out = Path(args.out)
    base = RunConfig(epochs=args.epochs, batch_size=args.batch_size)
    archs = [_parse_arch(a) for a in args.architectures]
    configs = [c for o in args.optimizers
               for c in build_grid(base, [o], [lrs[o]], archs, list(range(args.seeds)))]
    print(f"Running {len(configs)} trainings (learning rates: {lrs})...")
    rows = run_grid(configs, data, out)

    summary = aggregate(rows)
    write_csv(summary, out / "arch_summary.csv")
    _save(plots.plot_architecture_study(summary), out / "architecture.png")


def cmd_train(args) -> None:
    data = _load(args)
    cfg = RunConfig(optimizer=args.optimizer, lr=args.lr, hidden=tuple(args.hidden), epochs=args.epochs,
                    batch_size=args.batch_size, dropout=args.dropout, seed=args.seed)
    print(f"Training {cfg}")
    model, hist = fit(cfg, data, verbose=True)
    test_loss, test_acc = evaluate(model, data.test)
    cm = confusion_matrix(model, data.test)
    per_class = (cm.diag().float() / cm.sum(1).clamp(min=1)).tolist()

    print(f"\nTest accuracy: {test_acc:.4f}   test loss: {test_loss:.4f}")
    print("Per-class test accuracy: " + "  ".join(f"{d}:{a:.3f}" for d, a in enumerate(per_class)))

    path = save_checkpoint(model, args.save, run_config=cfg.as_dict(), test_acc=test_acc,
                           val_acc=hist.val_acc[-1])
    print(f"Saved checkpoint to {path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"config": cfg.as_dict(), "params": model.num_parameters(), "test_acc": test_acc,
               "test_loss": test_loss, "per_class_test_acc": per_class,
               "history": {"train_loss": hist.train_loss, "val_loss": hist.val_loss,
                           "val_acc": hist.val_acc}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save(plots.plot_confusion_matrix(cm.tolist()), out / "confusion_matrix.png")


def cmd_evaluate(args) -> None:
    model, meta = load_checkpoint(args.checkpoint)
    data = _load(args)
    model.to(data.test.x.device)
    test_loss, test_acc = evaluate(model, data.test)
    cm = confusion_matrix(model, data.test)
    print(f"Checkpoint: {args.checkpoint}  ({model.num_parameters():,} parameters)")
    if meta.get("run_config"):
        print(f"Trained with: {meta['run_config']}")
    print(f"Test accuracy: {test_acc:.4f}   test loss: {test_loss:.4f}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _save(plots.plot_confusion_matrix(cm.tolist()), out / "confusion_matrix.png")


# ----------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="data", help="where MNIST is downloaded")
    common.add_argument("--val-size", type=int, default=10_000, help="images held out for validation")
    common.add_argument("--limit-train", type=int, default=None, help="use only N training images (quick runs)")
    common.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    common.add_argument("--batch-size", type=int, default=128)

    parser = argparse.ArgumentParser(prog="python -m optimstudy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    names = list(OPTIMIZERS)

    p = sub.add_parser("sweep", parents=[common], help="learning-rate sweep")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--optimizers", nargs="+", choices=names, default=["sgd", "momentum", "rmsprop", "adam"])
    p.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    p.add_argument("--hidden", nargs="*", type=int, default=[128])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/sweep")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("arch", parents=[common], help="model size / depth study")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--optimizers", nargs="+", choices=names, default=["momentum", "adam"])
    p.add_argument("--architectures", nargs="+", default=DEFAULT_ARCHS,
                   help='hidden sizes per architecture, e.g. "256,128" (use "none" for no hidden layer)')
    p.add_argument("--best", help="best.json from a sweep, to reuse each optimizer's best learning rate")
    p.add_argument("--lr", type=float, help="learning rate for optimizers not found in --best")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="results/arch")
    p.set_defaults(func=cmd_arch)

    p = sub.add_parser("train", parents=[common], help="train one model and save it")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--optimizer", choices=names, default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", nargs="*", type=int, default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="checkpoints/mnist_mlp.pt")
    p.add_argument("--out", default="results/final")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", parents=[common], help="evaluate a saved checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--out", default=None, help="folder for the confusion matrix plot")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
