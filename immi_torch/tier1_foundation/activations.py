"""
Module 02: Activations - Enabling Non-Linear Learning

Non-linear functions applied element-wise to tensors.
"""

import numpy as np
from .tensor import Tensor


class ReLU:
    """Rectified Linear Unit activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data)
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Sigmoid:
    """Sigmoid activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        z = np.clip(x.data, -500, 500)
        result = np.zeros_like(z)
        
        # Positive values:
        z_pos = z >= 0
        result[z_pos] = 1.0 / (1.0 + np.exp(-z[z_pos]))
        
        # Negative values:
        z_neg = z < 0
        exp_n = np.exp(z[z_neg])
        result[z_neg] = exp_n / (1.0 + exp_n)
        
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Tanh:
    """Hyperbolic tangent activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class GELU:
    """Gaussian Error Linear Unit activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result = sigmoid_part * x.data
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Softmax:
    """Softmax activation function."""
    
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        # softmax(x) = exp(x - C) / sum(exp(x - C)) to prevent overflow (C = max(x))
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max
        
        exp_values = np.exp(x_shifted)
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)
        
        return Tensor(exp_values / exp_sum)
    
    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)
        
        