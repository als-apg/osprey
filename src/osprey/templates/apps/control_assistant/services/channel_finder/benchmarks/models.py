"""
Data models for benchmark system.

Defines all dataclasses used for benchmark entries, results, and evaluation metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class QueryBenchmarkEntry:
    """Single benchmark entry from dataset."""
    user_query: str
    targeted_pv: List[str]
    details: Optional[str] = None


@dataclass
class QueryRunResult:
    """Result from a single run of a query."""
    run_number: int
    found_pvs: List[str]
    success: bool
    error: Optional[str] = None
    execution_time_seconds: float = 0.0


@dataclass
class QueryEvaluation:
    """Evaluation metrics for a single query across all runs."""
    query_index: int
    user_query: str
    expected_pvs: List[str]
    details: Optional[str]
    
    # Aggregate across all runs
    total_runs: int
    successful_runs: int
    failed_runs: int
    
    # Best run metrics (run with highest F1 score)
    best_run_number: int
    best_found_pvs: List[str]
    best_true_positives: int
    best_false_positives: int
    best_false_negatives: int
    best_precision: float
    best_recall: float
    best_f1_score: float
    
    # Average metrics across all successful runs
    avg_precision: float
    avg_recall: float
    avg_f1_score: float
    
    # Consistency metrics
    consistency_score: float  # How similar are results across runs?
    unique_results_count: int  # How many different result sets?
    
    # Individual run details
    runs: List[QueryRunResult]


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    benchmark_name: str
    timestamp: str
    config_snapshot: Dict[str, Any]
    
    # Dataset info
    total_queries: int
    queries_evaluated: int
    
    # Aggregate metrics
    overall_precision: float
    overall_recall: float
    overall_f1_score: float
    
    # Success rate
    perfect_matches: int  # Queries with F1 = 1.0
    partial_matches: int  # Queries with 0 < F1 < 1.0
    no_matches: int  # Queries with F1 = 0.0
    
    # Per-query results
    query_evaluations: List[QueryEvaluation]
    
    # Summary statistics
    avg_consistency_score: float
    avg_execution_time: float

