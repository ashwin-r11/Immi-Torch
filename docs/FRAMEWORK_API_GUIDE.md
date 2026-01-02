# 🔥 Immi-Torch: From Project to Framework

> A step-by-step guide to transforming your educational ML library into a proper Python framework with a clean, stable API.

---

## Table of Contents

1. [What Makes a "Framework"?](#1-what-makes-a-framework)
2. [Understanding Your Current Structure](#2-understanding-your-current-structure)
3. [Designing the Public API](#3-designing-the-public-api)
4. [Implementation Steps](#4-implementation-steps)
5. [Packaging & Dependencies](#5-packaging--dependencies)
6. [Documentation & Discoverability](#6-documentation--discoverability)
7. [Versioning & Stability](#7-versioning--stability)
8. [Testing Your API](#8-testing-your-api)
9. [Checklist](#9-checklist)

---

## 1. What Makes a "Framework"?

### The Difference: Library vs Framework vs API

| Concept | Definition | Example |
|---------|------------|---------|
| **Library** | Collection of functions/classes you call | NumPy, Requests |
| **Framework** | Opinionated structure that calls YOUR code | Django, PyTorch |
| **API** | The "contract" - what users can reliably import and use | `torch.nn.Linear` |

### Key Characteristics of a Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAMEWORK ANATOMY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │  Public API  │     │  Internals   │     │   Plugins/   │   │
│   │  (Stable)    │────▶│  (Can Change)│◀────│   Extensions │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                     │                    │            │
│         ▼                     ▼                    ▼            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Documentation                         │  │
│   │         (Users only see the public surface)              │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Users Expect from a Framework

1. **Predictable imports** - `from framework import Thing` always works
2. **Stable API** - Code written today works tomorrow
3. **Clear documentation** - What's public vs internal
4. **Semantic versioning** - Breaking changes = major version bump
5. **Sensible defaults** - Works out of the box

---

## 2. Understanding Your Current Structure

### Your Tier-Based Layout (Educational)

```
immi_torch/
├── __init__.py                 # Root package
├── tier1_foundation/           # Core primitives
│   ├── tensor.py              # Tensor class
│   ├── autograd.py            # Automatic differentiation
│   ├── layers.py              # Linear, Module base
│   ├── activations.py         # ReLU, Sigmoid, etc.
│   ├── losses.py              # MSE, CrossEntropy
│   ├── optim.py               # SGD, Adam
│   ├── data.py                # DataLoader, Dataset
│   └── train.py               # Training utilities
├── tier2_architecture/         # Model components
│   ├── attention.py           # Attention mechanisms
│   ├── transformer.py         # Transformer blocks
│   ├── cnn.py                 # Conv layers
│   ├── embeddings.py          # Embedding layers
│   └── tokenizer.py           # Tokenization
└── tier3_optimization/         # Performance
    ├── quantization.py        # Model quantization
    ├── profiling.py           # Performance profiling
    └── ...
```

### The Problem with Current Structure

Right now, users must know your internal organization:

```python
# Current (leaky abstraction - exposes internals)
from immi_torch.tier1_foundation.tensor import Tensor
from immi_torch.tier1_foundation.layers import Linear
from immi_torch.tier2_architecture.transformer import Transformer

# Users shouldn't need to know about "tiers"!
```

### The Goal: Clean Public API

```python
# Goal (clean framework API)
from immi_torch import Tensor
from immi_torch.nn import Linear, ReLU, Transformer
from immi_torch.optim import SGD, Adam
from immi_torch.data import DataLoader, Dataset

# Much cleaner! Tiers are implementation details.
```

---

## 3. Designing the Public API

### API Design Principles

1. **Minimize Surface Area** - Expose only what users need
2. **Consistency** - Similar things should look similar
3. **Discoverability** - Easy to find what you need
4. **Backward Compatibility** - Don't break existing code

### Proposed Namespace Structure

```
immi_torch                     # Root namespace
├── Tensor                     # Core data structure (direct export)
├── nn/                        # Neural network components
│   ├── Module                 # Base class for all layers
│   ├── Linear                 # Fully connected layer
│   ├── Conv2d                 # Convolutional layer
│   ├── ReLU, Sigmoid, Tanh    # Activations
│   ├── Softmax                # Softmax activation
│   ├── Embedding              # Embedding layer
│   ├── MultiHeadAttention     # Attention mechanism
│   └── Transformer            # Transformer block
├── optim/                     # Optimizers
│   ├── Optimizer              # Base class
│   ├── SGD                    # Stochastic Gradient Descent
│   └── Adam                   # Adam optimizer
├── data/                      # Data handling
│   ├── Dataset                # Base dataset class
│   └── DataLoader             # Batching and iteration
├── autograd/                  # Automatic differentiation
│   └── backward()             # Backpropagation
├── losses/                    # Loss functions
│   ├── MSELoss                # Mean Squared Error
│   └── CrossEntropyLoss       # Cross Entropy
└── functional/                # Stateless functions (optional)
    ├── relu()
    ├── softmax()
    └── ...
```

### Mapping Tiers to Public API

| Internal (Tier) | Public API | Why |
|-----------------|------------|-----|
| `tier1_foundation.tensor.Tensor` | `immi_torch.Tensor` | Core type, top-level |
| `tier1_foundation.layers.Linear` | `immi_torch.nn.Linear` | Neural network component |
| `tier1_foundation.activations.ReLU` | `immi_torch.nn.ReLU` | Neural network component |
| `tier1_foundation.optim.SGD` | `immi_torch.optim.SGD` | Optimizer |
| `tier2_architecture.transformer` | `immi_torch.nn.Transformer` | Neural network component |

---

## 4. Implementation Steps

### Step 1: Create Facade Packages

A "facade" is a design pattern - a simple interface to a complex subsystem.

**Create `immi_torch/nn/__init__.py`:**

```python
"""
immi_torch.nn - Neural Network Building Blocks

This module provides the core components for building neural networks:
- Layer classes (Linear, Conv2d, etc.)
- Activation functions (ReLU, Sigmoid, etc.)
- Container modules (Sequential)

Example:
    >>> import immi_torch.nn as nn
    >>> layer = nn.Linear(784, 128)
    >>> activation = nn.ReLU()
"""

# Import from internal tier modules
from ..tier1_foundation.layers import Module, Linear
from ..tier1_foundation.activations import ReLU, Sigmoid, Tanh, Softmax

# Import from tier2 (architecture components)
from ..tier2_architecture.attention import MultiHeadAttention
from ..tier2_architecture.transformer import TransformerBlock
from ..tier2_architecture.cnn import Conv2d
from ..tier2_architecture.embeddings import Embedding

# Define what's public (controls `from immi_torch.nn import *`)
__all__ = [
    # Base
    "Module",
    # Layers
    "Linear",
    "Conv2d",
    "Embedding",
    # Activations
    "ReLU",
    "Sigmoid", 
    "Tanh",
    "Softmax",
    # Advanced
    "MultiHeadAttention",
    "TransformerBlock",
]
```

**Create `immi_torch/optim/__init__.py`:**

```python
"""
immi_torch.optim - Optimization Algorithms

This module provides optimizers for training neural networks.

Example:
    >>> import immi_torch.optim as optim
    >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
    >>> optimizer.step()
"""

from ..tier1_foundation.optim import Optimizer, SGD, Adam

__all__ = ["Optimizer", "SGD", "Adam"]
```

**Create `immi_torch/data/__init__.py`:**

```python
"""
immi_torch.data - Data Loading Utilities

This module provides tools for loading and batching data.

Example:
    >>> from immi_torch.data import DataLoader, Dataset
    >>> loader = DataLoader(dataset, batch_size=32)
    >>> for batch in loader:
    ...     # process batch
"""

from ..tier1_foundation.data import Dataset, DataLoader

__all__ = ["Dataset", "DataLoader"]
```

**Create `immi_torch/losses/__init__.py`:**

```python
"""
immi_torch.losses - Loss Functions

Loss functions for training neural networks.

Example:
    >>> from immi_torch.losses import MSELoss, CrossEntropyLoss
    >>> criterion = MSELoss()
    >>> loss = criterion(predictions, targets)
"""

from ..tier1_foundation.losses import MSELoss, CrossEntropyLoss

__all__ = ["MSELoss", "CrossEntropyLoss"]
```

### Step 2: Update Root `__init__.py`

**Update `immi_torch/__init__.py`:**

```python
"""
Immi-Torch: An Educational Deep Learning Framework
==================================================

Immi-Torch is a from-scratch implementation of a deep learning framework,
built for learning and understanding the internals of ML systems.

Quick Start:
    >>> from immi_torch import Tensor
    >>> from immi_torch import nn, optim
    >>> 
    >>> # Create a simple model
    >>> model = nn.Linear(784, 10)
    >>> optimizer = optim.SGD([model.weight], lr=0.01)

Submodules:
    - nn: Neural network layers and activations
    - optim: Optimization algorithms  
    - data: Data loading utilities
    - losses: Loss functions
    - autograd: Automatic differentiation

For more information, visit: https://github.com/ashwin-r11/Immi-Torch
"""

__version__ = "0.1.0"

# Core type - always available at top level
from .tier1_foundation.tensor import Tensor

# Submodules - import to make them accessible
from . import nn
from . import optim
from . import data
from . import losses

# For `from immi_torch import *`
__all__ = [
    "Tensor",
    "nn",
    "optim", 
    "data",
    "losses",
    "__version__",
]
```

### Step 3: Create the Directory Structure

You need to create these new directories/files:

```bash
# From your project root, run:
mkdir -p immi_torch/nn
mkdir -p immi_torch/optim  
mkdir -p immi_torch/data
mkdir -p immi_torch/losses

# Create the __init__.py files
touch immi_torch/nn/__init__.py
touch immi_torch/optim/__init__.py
touch immi_torch/data/__init__.py
touch immi_torch/losses/__init__.py
```

### Step 4: Handling Import Conflicts

**Problem:** You already have `immi_torch/tier1_foundation/optim.py` and now you're creating `immi_torch/optim/` package.

**Solution:** The new `immi_torch/optim/` package is a facade that RE-EXPORTS from the tier module. They coexist because they're at different paths:

```
immi_torch/tier1_foundation/optim.py  ← Implementation (internal)
immi_torch/optim/__init__.py          ← Public API (facade)
```

---

## 5. Packaging & Dependencies

### Update `pyproject.toml`

Your current `pyproject.toml` likely needs these changes:

```toml
[project]
name = "immi-torch"
version = "0.1.0"
description = "An educational deep learning framework built from scratch"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"

# IMPORTANT: Add required dependencies (not optional!)
dependencies = [
    "numpy>=1.24.0",
]

# Keep dev tools as optional
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/ashwin-r11/Immi-Torch"
Documentation = "https://github.com/ashwin-r11/Immi-Torch#readme"
Repository = "https://github.com/ashwin-r11/Immi-Torch"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["immi_torch*"]
```

### Why Dependencies Matter

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY TYPES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Required Dependencies (always installed)                       │
│   ├── numpy        ← Your Tensor depends on this!               │
│   └── ...                                                        │
│                                                                  │
│   Optional Dependencies (only when specified)                    │
│   ├── dev          ← pytest, black, etc.                        │
│   ├── docs         ← sphinx, mkdocs, etc.                       │
│   └── gpu          ← cupy (future GPU support)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Installing Your Package

```bash
# Install in development mode (editable)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Now users can do:
# pip install immi-torch
```

---

## 6. Documentation & Discoverability

### The `__all__` Variable

`__all__` is a list that defines what's "public" in a module:

```python
# In immi_torch/nn/__init__.py
__all__ = ["Module", "Linear", "ReLU"]

# This controls two things:
# 1. What `from immi_torch.nn import *` imports
# 2. What documentation tools consider "public"
```

### Docstring Standards

Use consistent docstrings for all public APIs:

```python
class Linear(Module):
    """Applies a linear transformation: y = xW^T + b
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If True, adds a learnable bias. Default: True
    
    Shape:
        - Input: (N, in_features) where N is batch size
        - Output: (N, out_features)
    
    Attributes:
        weight (Tensor): Learnable weights of shape (out_features, in_features)
        bias (Tensor): Learnable bias of shape (out_features)
    
    Example:
        >>> layer = Linear(20, 30)
        >>> x = Tensor(np.random.randn(128, 20))
        >>> output = layer(x)
        >>> print(output.shape)
        (128, 30)
    """
```

### README Structure for Frameworks

Update your `README.md` to include:

```markdown
# 🔥 Immi-Torch

An educational deep learning framework built from scratch.

## Installation

```bash
pip install immi-torch
```

## Quick Start

```python
from immi_torch import Tensor
from immi_torch import nn, optim

# Create data
x = Tensor([[1, 2], [3, 4], [5, 6]])
y = Tensor([[1], [2], [3]])

# Build model
model = nn.Linear(2, 1)

# Train
optimizer = optim.SGD([model.weight, model.bias], lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## API Reference

### Core
- `immi_torch.Tensor` - Multi-dimensional array with autodiff support

### Neural Networks (`immi_torch.nn`)
- `nn.Module` - Base class for all neural network modules
- `nn.Linear` - Fully connected layer
- `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh` - Activation functions

### Optimizers (`immi_torch.optim`)
- `optim.SGD` - Stochastic Gradient Descent
- `optim.Adam` - Adam optimizer

### Data (`immi_torch.data`)
- `data.Dataset` - Abstract dataset class
- `data.DataLoader` - Batching and iteration
```

---

## 7. Versioning & Stability

### Semantic Versioning (SemVer)

```
MAJOR.MINOR.PATCH
  │     │     │
  │     │     └── Bug fixes (backward compatible)
  │     └──────── New features (backward compatible)
  └────────────── Breaking changes (NOT backward compatible)
```

**Examples:**
- `0.1.0` → `0.1.1`: Fixed a bug
- `0.1.1` → `0.2.0`: Added new `nn.LayerNorm` class
- `0.2.0` → `1.0.0`: Changed `Tensor.backward()` signature

### Stability Guarantees

Add this to your README:

```markdown
## Stability

Immi-Torch follows semantic versioning.

### Public API (Stable)
These imports are stable and won't break without a major version bump:
- `immi_torch.Tensor`
- `immi_torch.nn.*`
- `immi_torch.optim.*`
- `immi_torch.data.*`

### Internal API (Unstable)
These may change at any time:
- `immi_torch.tier1_foundation.*`
- `immi_torch.tier2_architecture.*`
- `immi_torch.tier3_optimization.*`

If you import from tier modules directly, your code may break on updates.
```

### Version Location

Store version in one place and reference it:

```python
# immi_torch/__init__.py
__version__ = "0.1.0"

# Access from code:
import immi_torch
print(immi_torch.__version__)  # "0.1.0"
```

---

## 8. Testing Your API

### Test That Imports Work

Create `tests/test_api.py`:

```python
"""Test that the public API is accessible."""

def test_top_level_imports():
    """Users should be able to import Tensor directly."""
    from immi_torch import Tensor
    assert Tensor is not None

def test_nn_imports():
    """Users should be able to import from nn submodule."""
    from immi_torch.nn import Module, Linear, ReLU
    from immi_torch import nn
    
    assert nn.Linear is Linear
    assert nn.ReLU is ReLU

def test_optim_imports():
    """Users should be able to import optimizers."""
    from immi_torch.optim import SGD, Adam
    from immi_torch import optim
    
    assert optim.SGD is SGD

def test_version():
    """Package should have a version."""
    import immi_torch
    assert hasattr(immi_torch, '__version__')
    assert isinstance(immi_torch.__version__, str)

def test_all_exports():
    """__all__ should be defined."""
    import immi_torch
    assert hasattr(immi_torch, '__all__')
    assert 'Tensor' in immi_torch.__all__
    assert 'nn' in immi_torch.__all__
```

### Run Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/

# Run only API tests
pytest tests/test_api.py -v
```

---

## 9. Checklist

### Directory Structure

- [ ] Create `immi_torch/nn/__init__.py`
- [ ] Create `immi_torch/optim/__init__.py`
- [ ] Create `immi_torch/data/__init__.py`
- [ ] Create `immi_torch/losses/__init__.py`

### Implementation

- [ ] Add imports and `__all__` to each facade module
- [ ] Update `immi_torch/__init__.py` with new structure
- [ ] Add `__version__` to root `__init__.py`
- [ ] Add docstrings to all facade modules

### Packaging

- [ ] Update `pyproject.toml` with required dependencies
- [ ] Test `pip install -e .` works
- [ ] Verify imports work after installation

### Documentation

- [ ] Update README with new import style
- [ ] Document public vs internal API
- [ ] Add version stability policy

### Testing

- [ ] Create `tests/test_api.py`
- [ ] Verify all public imports work
- [ ] Run full test suite

---

## Final Structure

After completing all steps, your project should look like:

```
immi_torch/
├── __init__.py                 # Root: exports Tensor, nn, optim, data
├── nn/
│   └── __init__.py            # Facade: exports layers, activations
├── optim/
│   └── __init__.py            # Facade: exports optimizers
├── data/
│   └── __init__.py            # Facade: exports data utilities
├── losses/
│   └── __init__.py            # Facade: exports loss functions
├── tier1_foundation/          # Internal implementation
│   ├── tensor.py
│   ├── layers.py
│   └── ...
├── tier2_architecture/        # Internal implementation
│   └── ...
└── tier3_optimization/        # Internal implementation
    └── ...
```

---

## Summary

| Before | After |
|--------|-------|
| `from immi_torch.tier1_foundation.tensor import Tensor` | `from immi_torch import Tensor` |
| `from immi_torch.tier1_foundation.layers import Linear` | `from immi_torch.nn import Linear` |
| `from immi_torch.tier1_foundation.optim import SGD` | `from immi_torch.optim import SGD` |
| Internal structure visible to users | Clean, PyTorch-like API |

**You've transformed a project into a framework!** 🎉

---

## Next Steps

1. **Follow this guide** step by step
2. **Test each step** before moving to the next
3. **Commit after each major step** (good Git hygiene!)
4. **Consider adding**:
   - `immi_torch.functional` for stateless ops
   - `immi_torch.utils` for helper functions
   - Type hints for better IDE support

Good luck! Feel free to ask questions as you implement. 🚀
