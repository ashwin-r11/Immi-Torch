"""Unit tests for transformer module."""
import pytest
from immi_torch.tier2_architecture import transformer

def test_transformer_import():
    assert hasattr(transformer, "__file__")