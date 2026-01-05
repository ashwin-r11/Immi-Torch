"""Unit tests for embeddings module."""
import pytest
from immi_torch.tier2_architecture import embeddings

def test_embeddings_import():
    assert hasattr(embeddings, "__file__")