"""Unit tests for data module."""
import pytest
from immi_torch.tier1_foundation import data

def test_data_import():
    assert hasattr(data, "__file__")