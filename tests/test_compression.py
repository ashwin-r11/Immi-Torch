"""Unit tests for compression module."""
import pytest
from immi_torch.tier3_optimization import compression

def test_compression_import():
    assert hasattr(compression, "__file__")