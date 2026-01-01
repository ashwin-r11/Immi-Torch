"""
Module 03: Layers - Building Blocks of Networks

Parameterized transformations that learn from data.
"""

# TODO: Implement layer classes
# - Module (base class)
# - Linear (fully-connected)
# - Parameter management
# - Weight initialization strategies


class Module:
    """Base class for all neural network modules."""
    
    def parameters(self):
        """Return all parameters of this module."""
        return []


class Linear(Module):
    """Fully-connected (dense) layer."""
    pass
