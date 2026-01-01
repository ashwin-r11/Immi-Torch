"""
Immi-Torch: Tier 3 - Optimization

Make it production-ready:
- Profiling: Find bottlenecks
- Quantization: Reduce precision
- Compression: Smaller models
- Memoization: Cache computations
- Acceleration: Hardware optimization
- Benchmarking: MLPerf-style metrics
"""

from .profiling import Profiler
from .quantization import quantize, dequantize
from .compression import prune, distill
from .memoization import cache, LRUCache
from .acceleration import compile, fuse_ops
from .benchmarking import Benchmark, mlperf_suite

__all__ = [
    "Profiler",
    "quantize", "dequantize",
    "prune", "distill",
    "cache", "LRUCache",
    "compile", "fuse_ops",
    "Benchmark", "mlperf_suite",
]
