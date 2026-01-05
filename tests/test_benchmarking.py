"""Unit tests for benchmarking module."""
import pytest
from immi_torch.tier3_optimization import benchmarking

def test_benchmarking_import():
    assert hasattr(benchmarking, "__file__")