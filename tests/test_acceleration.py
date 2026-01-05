"""Unit tests for acceleration module."""
import pytest
from immi_torch.tier3_optimization import acceleration

def test_acceleration_import():
    assert hasattr(acceleration, "__file__")