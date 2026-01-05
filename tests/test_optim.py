"""Unit tests for optim module."""
import pytest
from immi_torch.tier1_foundation import optim

def test_optim_import():
    assert hasattr(optim, "__file__")