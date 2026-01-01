"""
Module 13: Attention - The Key Innovation

Self-attention mechanism that enables transformers.
"""

# TODO: Implement attention mechanisms
# - Scaled dot-product attention
# - Multi-head attention
# - Causal masking for autoregressive models


class Attention:
    """Scaled dot-product attention."""
    
    def forward(self, query, key, value, mask=None):
        """Compute attention weights and output."""
        pass


class MultiHeadAttention:
    """Multi-head attention layer."""
    
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
