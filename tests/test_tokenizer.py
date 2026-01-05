"""Unit tests for tokenizer module."""
import pytest
from immi_torch.tier2_architecture import tokenizer

def test_tokenizer_import():
    assert hasattr(tokenizer, "__file__")