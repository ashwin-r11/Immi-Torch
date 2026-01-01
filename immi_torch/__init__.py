"""
Immi-Torch: A minimal deep learning framework built from scratch.

"Immi" (Tamil: இம்மி) denotes a tiny fraction (1/2,150,400) — the smallest primitive measure.

Structure:
- tier1_foundation: Core ML machinery (Modules 01-08)
- tier2_architecture: Vision & Language tracks (Modules 09-14)
- tier3_optimization: Production optimization (Modules 15-20)
"""

__version__ = "0.1.0"

# Tier 1: Foundation
from .tier1_foundation import (
    Tensor,
    ReLU, Sigmoid, Tanh, Softmax,
    Linear, Module,
    MSELoss, CrossEntropyLoss,
    DataLoader, Dataset,
    backward,
    SGD, Adam, RMSprop,
    Trainer,
)

# Tier 2: Architecture (import when needed)
# from .tier2_architecture import Conv2d, Transformer, GPT, ...

# Tier 3: Optimization (import when needed)
# from .tier3_optimization import Profiler, quantize, ...

__all__ = [
    # Tier 1 exports
    "Tensor",
    "ReLU", "Sigmoid", "Tanh", "Softmax",
    "Linear", "Module",
    "MSELoss", "CrossEntropyLoss",
    "DataLoader", "Dataset",
    "backward",
    "SGD", "Adam", "RMSprop",
    "Trainer",
]
