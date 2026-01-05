"""Unit tests for attention module."""
import pytest
from immi_torch.tier2_architecture import attention

def test_attention_import():
    assert hasattr(attention, "__file__")