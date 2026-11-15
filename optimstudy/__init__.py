"""How much does the choice of optimizer, learning rate and model size matter for an MLP on MNIST?"""

from .checkpoint import load_checkpoint, save_checkpoint
from .data import Data, Split, load_mnist
from .model import MLP
from .optimizers import OPTIMIZERS, make_optimizer
from .training import RunConfig, evaluate, fit

__version__ = "0.1.0"

__all__ = [
    "MLP", "OPTIMIZERS", "make_optimizer", "RunConfig", "fit", "evaluate",
    "load_mnist", "Data", "Split", "save_checkpoint", "load_checkpoint",
]
