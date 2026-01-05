"""Unit tests for cnn module."""
import pytest
from immi_torch.tier2_architecture import cnn

def test_cnn_import():
    assert hasattr(cnn, "__file__")