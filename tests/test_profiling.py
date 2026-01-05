"""Unit tests for profiling module."""
import pytest
from immi_torch.tier3_optimization import profiling

def test_profiling_import():
    assert hasattr(profiling, "__file__")