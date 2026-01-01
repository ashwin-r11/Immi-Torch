# Examples

Example scripts demonstrating Immi-Torch usage.

## Quick Start

```python
from immi_torch import Tensor, Linear, ReLU, MSELoss, SGD

# Create a simple model
model = Linear(10, 1)
activation = ReLU()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Forward pass
x = Tensor.randn(32, 10)
y = Tensor.randn(32, 1)

pred = activation(model(x))
loss = loss_fn(pred, y)

# Backward pass
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
