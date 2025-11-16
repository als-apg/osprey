"""
Channel Finder Benchmarks (Core)

A comprehensive benchmarking system for evaluating channel finder performance.
Works with any pipeline implementation (in_context, hierarchical, etc.).

Includes dataset management, statistical analysis, and result reporting.

Usage:
    # Via main CLI (recommended)
    channel-finder benchmark
    channel-finder benchmark --example hierarchical --queries 0:10
    
    # Or programmatically
    from channel_finder.benchmarks import BenchmarkRunner
    runner = BenchmarkRunner()
    await runner.run_all_enabled_benchmarks()
"""

from .models import (
    QueryBenchmarkEntry,
    QueryRunResult,
    QueryEvaluation,
    BenchmarkResults
)
from .runner import BenchmarkRunner

__all__ = [
    'QueryBenchmarkEntry',
    'QueryRunResult', 
    'QueryEvaluation',
    'BenchmarkResults',
    'BenchmarkRunner',
]

__version__ = '1.0.0'

