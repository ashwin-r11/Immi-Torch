"""
Module 07: Optimizers - Learning from Gradients

Algorithms that update parameters using gradients.
"""

# TODO: Implement optimizers
# - SGD (with momentum)
# - Adam
# - RMSprop
# - Learning rate scheduling


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
    
    def step(self):
        """Update parameters."""
        pass
    
    def zero_grad(self):
        """Reset gradients to zero."""
        pass


class Adam:
    """Adam optimizer."""
    pass


class RMSprop:
    """RMSprop optimizer."""
    pass
