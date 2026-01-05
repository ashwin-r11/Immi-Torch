"""Unit tests for train module."""
import pytest
from immi_torch.tier1_foundation import train

def test_train_import():
    assert hasattr(train, "__file__")