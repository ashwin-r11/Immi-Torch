"""
Immi-Torch: Tier 1 - Foundation

Core machinery for deep learning:
- Tensor: Data structure
- Activations: Non-linearity
- Layers: Parameterized transformations
- Losses: Error measurement
- Data: DataLoader & Dataset
- Autograd: Gradient computation
- Optim: Optimizers
- Train: Training loop
"""

from .tensor import Tensor
from .activations import ReLU, Sigmoid, Tanh, Softmax, GELU
from .layers import Linear, Module
from .losses import MSELoss, CrossEntropyLoss
from .data import DataLoader, Dataset
from .autograd import backward
from .optim import SGD, Adam, RMSprop
from .train import Trainer

__all__ = [
    "Tensor",
    "ReLU", "Sigmoid", "Tanh", "Softmax", "GELU",
    "Linear", "Module",
    "MSELoss", "CrossEntropyLoss",
    "DataLoader", "Dataset",
    "backward",
    "SGD", "Adam", "RMSprop",
    "Trainer",
]
