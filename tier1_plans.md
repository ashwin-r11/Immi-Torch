# Immi-Torch: Tier 1 - Foundation

> I'm building a deep learning framework from scratch, understanding every component.

---

## Phase 1: Forward Pass Components (01-04)

**Tensors (01) → Activations (02) → Layers (03) → Losses (04)**

I must build things in the order data flows through them:

- **Tensors** are the data structure—I can't do anything without them
- **Activations** transform tensors non-linearly—needed before layers can create interesting functions
- **Layers** combine tensors and activations into parameterized transformations
- **Losses** measure how wrong predictions are—needed before I can learn

> At this point, I can do a complete forward pass: `input → layer → activation → loss`

---

## Phase 2: Learning Infrastructure (05-07)

**DataLoader (05) → Autograd (06) → Optimizers (07)**

Now I need the infrastructure to learn from data:

5. **DataLoader** provides efficient data batching—real training needs this before autograd
6. **Autograd** computes gradients automatically—the engine that makes learning possible
7. **Optimizers** use gradients to update parameters—SGD, Adam, and friends

---

## Phase 3: Complete Training (08)

**Training (08)** integrates everything into a complete learning loop.

> This order isn't arbitrary—it's the minimal dependency chain. I can't build optimizers without autograd (no gradients), can't build autograd without losses (nothing to differentiate), can't build losses without layers (no predictions). Each module unlocks the next.

---

## Module Details

### 01. Tensor - The Foundation of Everything

| | |
|---|---|
| **What it is** | Multidimensional arrays with automatic shape tracking and broadcasting. |
| **Why it matters** | Tensors are the universal data structure for ML. Understanding tensor operations, broadcasting, and memory layouts is essential for building efficient neural networks. |
| **What I'll build** | A pure Python tensor class supporting arithmetic, reshaping, slicing, and broadcasting—just like PyTorch tensors. |
| **Systems focus** | Memory layout, broadcasting semantics, operation fusion |

---

### 02. Activations - Enabling Non-Linear Learning

| | |
|---|---|
| **What it is** | Non-linear functions applied element-wise to tensors. |
| **Why it matters** | Without activations, neural networks collapse to linear models. Activations like ReLU, Sigmoid, and Tanh enable networks to learn complex, non-linear patterns. |
| **What I'll build** | Common activation functions with their gradients for backpropagation. |
| **Systems focus** | Numerical stability, in-place operations, gradient flow |

---

### 03. Layers - Building Blocks of Networks

| | |
|---|---|
| **What it is** | Parameterized transformations (Linear, Conv2d) that learn from data. |
| **Why it matters** | Layers are the modular components you stack to build networks. Understanding weight initialization, parameter management, and forward passes is crucial. |
| **What I'll build** | Linear (fully-connected) layers with proper initialization and parameter tracking. |
| **Systems focus** | Parameter storage, initialization strategies, forward computation |

---

### 04. Losses - Measuring Success

| | |
|---|---|
| **What it is** | Functions that quantify how wrong my predictions are. |
| **Why it matters** | Loss functions define what "good" means for my model. Different tasks (classification, regression) require different loss functions. |
| **What I'll build** | CrossEntropyLoss, MSELoss, and other common objectives with their gradients. |
| **Systems focus** | Numerical stability (log-sum-exp trick), reduction strategies |

---

### 05. DataLoader - Efficient Data Pipelines

| | |
|---|---|
| **What it is** | Infrastructure for loading, batching, and shuffling training data efficiently. |
| **Why it matters** | Real ML systems train on datasets that don't fit in memory. DataLoaders handle batching, shuffling, and parallel data loading, which are essential for efficient training. |
| **What I'll build** | A DataLoader that supports batching, shuffling, and dataset iteration with proper memory management. |
| **Systems focus** | Memory efficiency, batching strategies, I/O optimization |

---

### 06. Autograd - The Gradient Revolution

| | |
|---|---|
| **What it is** | Automatic differentiation system that computes gradients through computation graphs. |
| **Why it matters** | Autograd is what makes deep learning practical. It automatically computes gradients for any computation, enabling backpropagation through arbitrarily complex networks. |
| **What I'll build** | A computational graph system that tracks operations and computes gradients via the chain rule. |
| **Systems focus** | Computational graphs, topological sorting, gradient accumulation |

---

### 07. Optimizers - Learning from Gradients

| | |
|---|---|
| **What it is** | Algorithms that update parameters using gradients (SGD, Adam, RMSprop). |
| **Why it matters** | Raw gradients don't directly tell me how to update parameters. Optimizers use momentum, adaptive learning rates, and other tricks to make training converge faster and more reliably. |
| **What I'll build** | SGD, Adam, and RMSprop with proper momentum and learning rate scheduling. |
| **Systems focus** | Update rules, momentum buffers, numerical stability |

---

### 08. Training - Orchestrating the Learning Process

| | |
|---|---|
| **What it is** | The training loop that ties everything together—forward pass, loss computation, backpropagation, parameter updates. |
| **Why it matters** | Training loops orchestrate the entire learning process. Understanding this flow—including batching, epochs, and validation—is essential for practical ML. |
| **What I'll build** | A complete training framework with progress tracking, validation, and model checkpointing. |
| **Systems focus** | Batch processing, gradient clipping, learning rate scheduling |

---

## What I Can Build After This Tier

### Historical Achievements Unlocked

| Year | Milestone | Description |
|------|-----------|-------------|
| **1957** | 🧠 Perceptron | Binary classification with gradient descent |
| **1969** | ⚡ XOR Crisis Solved | Hidden layers enable non-linear learning |
| **1986** | 🚀 MLP Revival | Multi-layer networks achieve 95%+ on MNIST |

> *Fig. 3: Foundation Tier Milestones. After completing modules 01-08, I unlock three historical achievements spanning three decades of neural network breakthroughs.*

After completing the Foundation tier, I'll be able to:

- **Milestone 01 (1957):** Recreate the Perceptron, the first trainable neural network
- **Milestone 02 (1969):** Solve the XOR problem that nearly ended AI research
- **Milestone 03 (1986):** Build multi-layer perceptrons that achieve 95%+ accuracy on MNIST

---

## My Prerequisites

### Required

- Python programming (functions, classes, loops)
- Basic linear algebra (matrix multiplication, dot products)
- Basic calculus (derivatives, chain rule)

### Helpful but not required imo

- NumPy experience
- Understanding of neural network concepts

---

## My Time Commitment

| | |
|---|---|
| **Per module** | 3-5 hours (implementation + exercises + systems thinking) |
| **Total tier** | ~25-35 hours for complete mastery |
| **My pace** | 1-2 modules per week |
