"""Channel Finder benchmarking system.

Usage::

    channel-finder benchmark
    channel-finder benchmark --example hierarchical --queries 0:10
"""

from .models import BenchmarkResults, QueryBenchmarkEntry, QueryEvaluation, QueryRunResult
from .runner import BenchmarkRunner

__all__ = [
    "QueryBenchmarkEntry",
    "QueryRunResult",
    "QueryEvaluation",
    "BenchmarkResults",
    "BenchmarkRunner",
]
