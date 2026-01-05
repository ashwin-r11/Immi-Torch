"""Unit tests for memoization module."""
import pytest
from immi_torch.tier3_optimization import memoization

def test_memoization_import():
    assert hasattr(memoization, "__file__")