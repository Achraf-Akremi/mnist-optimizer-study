"""Fast tests on synthetic MNIST-shaped data (no download needed)."""

import json

import pytest
import torch

from optimstudy import MLP, OPTIMIZERS, RunConfig, fit, load_checkpoint, make_optimizer, save_checkpoint
from optimstudy import cli
from optimstudy.data import Data, Split, batches
from optimstudy.experiments import aggregate, best_per_optimizer, build_grid
from optimstudy.training import confusion_matrix, evaluate


def synthetic_split(n: int, seed: int) -> Split:
    """Random images where the label is encoded in one bright pixel, so it's learnable."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.randn(n, 784, generator=g) * 0.5
    x[torch.arange(n), y * 70] += 4.0
    return Split(x, y)


@pytest.fixture
def data() -> Data:
    return Data(synthetic_split(1200, 0), synthetic_split(300, 1), synthetic_split(300, 2))


def test_mlp_shapes_and_params():
    model = MLP((64, 32))
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)
    assert model.num_parameters() == 784 * 64 + 64 + 64 * 32 + 32 + 32 * 10 + 10
    assert MLP(()).num_parameters() == 784 * 10 + 10  # logistic regression


def test_every_optimizer_can_be_built():
    for name in OPTIMIZERS:
        assert isinstance(make_optimizer(name, MLP((8,)).parameters(), 1e-3), torch.optim.Optimizer)
    with pytest.raises(ValueError):
        make_optimizer("nope", MLP((8,)).parameters(), 1e-3)


def test_batches_cover_split_exactly_once():
    split = synthetic_split(103, 0)
    seen = torch.cat([y for _, y in batches(split, 10, shuffle=True, generator=torch.Generator().manual_seed(0))])
    assert len(seen) == 103
    assert torch.equal(seen.sort().values, split.y.sort().values)


def test_training_learns_and_is_reproducible(data):
    cfg = RunConfig(optimizer="adam", lr=3e-3, hidden=(32,), epochs=4, batch_size=64, seed=7)
    model, hist = fit(cfg, data)
    assert hist.train_loss[-1] < hist.train_loss[0]
    assert hist.val_acc[-1] > 0.9
    _, hist2 = fit(cfg, data)
    assert hist.val_loss == pytest.approx(hist2.val_loss)


def test_divergence_is_detected(data):
    cfg = RunConfig(optimizer="sgd", lr=1e6, hidden=(32,), epochs=5, batch_size=64)
    _, hist = fit(cfg, data)
    assert hist.diverged and len(hist.train_loss) < 5


def test_accuracy_and_confusion_matrix_agree(data):
    model, _ = fit(RunConfig(hidden=(16,), epochs=1, batch_size=64), data)
    _, acc = evaluate(model, data.test)
    cm = confusion_matrix(model, data.test)
    assert cm.sum().item() == len(data.test)
    assert cm.diag().sum().item() / len(data.test) == pytest.approx(acc)


def test_checkpoint_roundtrip(tmp_path, data):
    model, _ = fit(RunConfig(hidden=(16, 8), epochs=1, dropout=0.1), data)
    path = save_checkpoint(model, tmp_path / "m.pt", note="hi")
    loaded, meta = load_checkpoint(path)
    model.eval()
    assert meta["note"] == "hi"
    assert torch.allclose(model(data.test.x), loaded(data.test.x))


def test_aggregate_and_best():
    rows = [
        {"optimizer": "adam", "lr": 1e-3, "hidden": "32", "params": 1, "final_val_acc": a,
         "final_val_loss": 0.1, "diverged": False}
        for a in (0.90, 0.92)
    ] + [
        {"optimizer": "adam", "lr": 1e-1, "hidden": "32", "params": 1, "final_val_acc": 0.5,
         "final_val_loss": 1.0, "diverged": False}
    ]
    summary = aggregate(rows)
    assert len(summary) == 2
    best = best_per_optimizer(summary)
    assert best["adam"]["lr"] == 1e-3
    assert best["adam"]["val_acc_mean"] == pytest.approx(0.91)
    assert len(build_grid(RunConfig(), ["sgd", "adam"], [1e-3, 1e-2], [(8,)], [0, 1, 2])) == 12


def test_cli_end_to_end(tmp_path, data, monkeypatch):
    monkeypatch.setattr(cli, "load_mnist", lambda **kw: data)
    common = ["--epochs", "2", "--device", "cpu"]

    cli.main(["sweep", *common, "--optimizers", "sgd", "adam", "--lrs", "1e-3", "1e-2",
              "--hidden", "16", "--seeds", "2", "--out", str(tmp_path / "sweep")])
    best_file = tmp_path / "sweep" / "best.json"
    assert set(json.loads(best_file.read_text())) == {"sgd", "adam"}
    for f in ("runs.csv", "summary.csv", "histories.json", "lr_sweep.png", "val_loss_curves.png"):
        assert (tmp_path / "sweep" / f).exists()

    cli.main(["arch", *common, "--optimizers", "sgd", "adam", "--architectures", "none", "16", "16,8",
              "--best", str(best_file), "--seeds", "1", "--out", str(tmp_path / "arch")])
    assert (tmp_path / "arch" / "architecture.png").exists()

    ckpt = tmp_path / "model.pt"
    cli.main(["train", *common, "--hidden", "16", "--save", str(ckpt), "--out", str(tmp_path / "final")])
    assert ckpt.exists() and (tmp_path / "final" / "confusion_matrix.png").exists()

    cli.main(["evaluate", str(ckpt), "--device", "cpu", "--out", str(tmp_path / "eval")])
    assert (tmp_path / "eval" / "confusion_matrix.png").exists()
