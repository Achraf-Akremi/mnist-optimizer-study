# MNIST Optimizer Study

How much do the **optimizer**, the **learning rate** and the **network size** matter when training a simple multilayer perceptron? This project answers that with a small, reproducible benchmark on MNIST, written in PyTorch.

It runs three experiments:

| Experiment | Question | Output |
|---|---|---|
| `sweep` | Which learning rate works best for SGD, SGD + momentum, RMSprop and Adam, and how sensitive is each one to it? | accuracy-vs-learning-rate curves, learning curves |
| `arch` | Does a wider or deeper network help, and does the answer depend on the optimizer? | accuracy by architecture |
| `train` / `evaluate` | How good is a final model on the untouched test set, and which digits does it confuse? | checkpoint, test accuracy, confusion matrix |

## Methodology

- **Proper data split.** 50,000 training images, 10,000 validation images (held out from the official training set), and the official 10,000 test images. All model selection uses the validation set; the test set is only used by `train` and `evaluate`.
- **Several seeds per configuration.** Every result is the mean ± standard deviation over independent runs (different initialization and data order), so differences caused by luck are visible.
- **Fair comparison.** Each optimizer is compared at *its own* best learning rate, found by the sweep.
- **Divergence detection.** Runs whose loss becomes NaN/inf are stopped early and flagged in the results.
- **Fast.** The dataset is kept in memory as normalized tensors instead of going through a per-image `DataLoader`, so a full sweep takes minutes on a laptop CPU.

## Project structure

```
mnist-optimizer-study/
├── optimstudy/
│   ├── data.py          # in-memory MNIST with train/val/test split
│   ├── model.py         # configurable MLP
│   ├── optimizers.py    # optimizers by name (sgd, momentum, nesterov, rmsprop, adam, adamw)
│   ├── training.py      # training loop, evaluation, confusion matrix
│   ├── experiments.py   # run grids, average over seeds, pick best settings
│   ├── checkpoint.py    # save / load trained models
│   ├── plots.py         # figures
│   └── cli.py           # `python -m optimstudy ...`
├── tests/               # fast tests on synthetic data (no download)
├── .github/workflows/   # runs the tests on every push
├── requirements.txt
└── requirements-dev.txt
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/mnist-optimizer-study.git
cd mnist-optimizer-study
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

MNIST (~50 MB) is downloaded into `data/` automatically on the first run.

## Usage

Run the three experiments in order. Each command prints a summary and writes CSV files and figures to `results/`.

```bash
# 1. Learning-rate sweep: 4 optimizers x 8 learning rates x 3 seeds
python -m optimstudy sweep

# 2. Architecture study, reusing each optimizer's best learning rate from step 1
python -m optimstudy arch --best results/sweep/best.json

# 3. Train a final model, evaluate it on the test set and save it
python -m optimstudy train --optimizer adam --lr 1e-3 --hidden 256 128 --epochs 10

# Evaluate a saved model later
python -m optimstudy evaluate checkpoints/mnist_mlp.pt --out results/eval
```

Useful options (see `python -m optimstudy <command> --help` for all of them):

```bash
python -m optimstudy sweep --optimizers sgd nesterov adamw --lrs 1e-3 1e-2 1e-1 --seeds 5
python -m optimstudy sweep --limit-train 5000 --seeds 1 --epochs 1   # quick try-out
python -m optimstudy arch --architectures none 64 "256,128" "512,256,128" --lr 1e-3
python -m optimstudy train --dropout 0.2 --device cuda
```

### Output files

| File | Contents |
|---|---|
| `results/sweep/runs.csv` | one row per training run |
| `results/sweep/summary.csv` | mean ± std over seeds per optimizer and learning rate |
| `results/sweep/best.json` | best learning rate per optimizer |
| `results/sweep/lr_sweep.png` | validation accuracy vs. learning rate |
| `results/sweep/val_loss_curves.png` | validation loss per epoch at each best learning rate |
| `results/arch/architecture.png` | validation accuracy per architecture and optimizer |
| `results/final/metrics.json` | test accuracy, per-class accuracy, training history |
| `results/final/confusion_matrix.png` | which digits get confused |
| `checkpoints/mnist_mlp.pt` | trained weights + architecture (not committed to git) |

### Using the package from Python

```python
from optimstudy import RunConfig, fit, evaluate, load_mnist, save_checkpoint

data = load_mnist()
model, history = fit(RunConfig(optimizer="momentum", lr=0.05, hidden=(256,), epochs=5), data)
print("val acc per epoch:", history.val_acc)
print("test acc:", evaluate(model, data.test)[1])
save_checkpoint(model, "checkpoints/momentum.pt")
```

## Tests

```bash
python -m pytest
```

The tests use synthetic data, so they run in under a minute without downloading MNIST. They check model shapes, batching, reproducibility, divergence detection, metric consistency, checkpoint round-trips and the full command-line pipeline.

## Results

After running the experiments, add your figures and findings here, for example:

```markdown
![Learning-rate sweep](results/sweep/lr_sweep.png)
![Architecture study](results/arch/architecture.png)
![Confusion matrix](results/final/confusion_matrix.png)
```
