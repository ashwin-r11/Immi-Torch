"""Unit tests for quantization module."""
import pytest
from immi_torch.tier3_optimization import quantization

def test_quantization_import():
    assert hasattr(quantization, "__file__")