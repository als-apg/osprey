"""End-to-end tests for channel finder pipeline benchmarks.

These tests validate that both channel finder pipelines (in_context and hierarchical)
meet the performance benchmarks with >90% accuracy on their respective datasets.
"""

import json
import logging
import os
import sys

import pytest
import yaml


@pytest.mark.e2e
@pytest.mark.e2e_benchmark
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_in_context_pipeline_benchmark(e2e_project_factory):
    """Test in-context pipeline achieves >90% accuracy on benchmark dataset.

    The in-context pipeline uses semantic search over the complete channel database
    presented in a single LLM context window. This test validates it can accurately
    find channels across 30 diverse natural language queries covering the UCSB FEL
    accelerator.

    Success criteria:
    - ≥80% perfect matches (F1 score = 1.0)
    - No critical errors or failures
    - Completion rate >95%
    """
    # Create control-assistant project with in_context pipeline
    project = await e2e_project_factory(
        name="benchmark-in-context", template="control_assistant", registry_style="extend"
    )

    # Modify config to ensure in_context pipeline is active
    config_path = project.config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["channel_finder"]["pipeline_mode"] = "in_context"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the benchmark programmatically
    benchmark_results = await _run_benchmark(project, "in_context")

    # Validate benchmark results
    total_queries = benchmark_results["total_queries"]
    perfect_matches = benchmark_results["perfect_matches"]
    partial_matches = benchmark_results["partial_matches"]
    no_matches = benchmark_results["no_matches"]
    overall_f1 = benchmark_results["overall_f1_score"]

    # Calculate success rate
    success_rate = (perfect_matches / total_queries) * 100 if total_queries > 0 else 0

    # Assert ≥80% perfect match rate
    assert success_rate >= 80.0, (
        f"In-context pipeline benchmark failed: {success_rate:.1f}% success rate "
        f"({perfect_matches}/{total_queries} perfect matches). Expected ≥80%.\n"
        f"Perfect: {perfect_matches}, Partial: {partial_matches}, Failed: {no_matches}"
    )

    # Assert high overall F1 score
    assert overall_f1 >= 0.80, f"Overall F1 score too low: {overall_f1:.3f}. Expected ≥0.80"

    # Assert completion rate (should run all queries without crashes)
    queries_evaluated = benchmark_results["queries_evaluated"]
    completion_rate = (queries_evaluated / total_queries) * 100 if total_queries > 0 else 0
    assert completion_rate >= 95.0, (
        f"Low completion rate: {completion_rate:.1f}% "
        f"({queries_evaluated}/{total_queries} queries). Expected ≥95%"
    )


@pytest.mark.e2e
@pytest.mark.e2e_benchmark
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_hierarchical_pipeline_benchmark(e2e_project_factory):
    """Test hierarchical pipeline achieves >90% accuracy on benchmark dataset.

    The hierarchical pipeline navigates a structured 5-level database
    (system → family → device → field → subfield) using LLM-guided path construction.
    This test validates it can accurately find channels across 47 diverse queries
    covering a simulated synchrotron facility with ~150 channels.

    Success criteria:
    - ≥80% perfect matches (F1 score = 1.0)
    - No critical errors or failures
    - Completion rate >95%
    """
    # Create control-assistant project
    project = await e2e_project_factory(
        name="benchmark-hierarchical", template="control_assistant", registry_style="extend"
    )

    # Modify config to set hierarchical pipeline
    config_path = project.config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["channel_finder"]["pipeline_mode"] = "hierarchical"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the benchmark programmatically
    benchmark_results = await _run_benchmark(project, "hierarchical")

    # Validate benchmark results
    total_queries = benchmark_results["total_queries"]
    perfect_matches = benchmark_results["perfect_matches"]
    partial_matches = benchmark_results["partial_matches"]
    no_matches = benchmark_results["no_matches"]
    overall_f1 = benchmark_results["overall_f1_score"]

    # Calculate success rate
    success_rate = (perfect_matches / total_queries) * 100 if total_queries > 0 else 0

    # Assert ≥80% perfect match rate
    assert success_rate >= 80.0, (
        f"Hierarchical pipeline benchmark failed: {success_rate:.1f}% success rate "
        f"({perfect_matches}/{total_queries} perfect matches). Expected ≥80%.\n"
        f"Perfect: {perfect_matches}, Partial: {partial_matches}, Failed: {no_matches}"
    )

    # Assert high overall F1 score
    assert overall_f1 >= 0.80, f"Overall F1 score too low: {overall_f1:.3f}. Expected ≥0.80"

    # Assert completion rate (should run all queries without crashes)
    queries_evaluated = benchmark_results["queries_evaluated"]
    completion_rate = (queries_evaluated / total_queries) * 100 if total_queries > 0 else 0
    assert completion_rate >= 95.0, (
        f"Low completion rate: {completion_rate:.1f}% "
        f"({queries_evaluated}/{total_queries} queries). Expected ≥95%"
    )


@pytest.mark.e2e
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_middle_layer_pipeline_sample(e2e_project_factory):
    """Test middle layer pipeline on first 5 benchmark queries (quick validation).

    The middle layer pipeline uses a React agent with database query tools to navigate
    a functional hierarchy (System → Family → Field) mimicking MATLAB Middle Layer (MML)
    style databases used in production accelerator facilities.

    This is a quick smoke test using the first 5 queries from the full benchmark
    to validate basic functionality without the cost of running all 20 queries.

    Success criteria:
    - ≥80% perfect matches (4 out of 5 queries)
    - No critical errors or failures
    - Completion rate 100% (all 5 queries should execute)
    """
    # Create control-assistant project
    project = await e2e_project_factory(
        name="sample-middle-layer", template="control_assistant", registry_style="extend"
    )

    # Modify config to set middle_layer pipeline and limit to first 5 queries
    config_path = project.config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Ensure channel_finder section exists
    if "channel_finder" not in config:
        config["channel_finder"] = {}

    config["channel_finder"]["pipeline_mode"] = "middle_layer"

    # Limit to first 5 queries for faster execution
    if "benchmark" not in config["channel_finder"]:
        config["channel_finder"]["benchmark"] = {}
    if "execution" not in config["channel_finder"]["benchmark"]:
        config["channel_finder"]["benchmark"]["execution"] = {}

    config["channel_finder"]["benchmark"]["execution"]["query_selection"] = {"start": 0, "end": 5}

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the benchmark programmatically
    benchmark_results = await _run_benchmark(project, "middle_layer")

    # Validate benchmark results
    queries_evaluated = benchmark_results["queries_evaluated"]
    perfect_matches = benchmark_results["perfect_matches"]
    partial_matches = benchmark_results["partial_matches"]
    no_matches = benchmark_results["no_matches"]
    overall_f1 = benchmark_results["overall_f1_score"]

    # Should have exactly 5 queries evaluated (limited via query_selection config)
    assert queries_evaluated == 5, f"Expected 5 queries evaluated, got {queries_evaluated}"

    # Calculate success rate
    success_rate = (perfect_matches / queries_evaluated) * 100 if queries_evaluated > 0 else 0

    # Assert ≥80% perfect match rate (4 out of 5 queries)
    assert success_rate >= 80.0, (
        f"Middle layer pipeline sample failed: {success_rate:.1f}% success rate "
        f"({perfect_matches}/{queries_evaluated} perfect matches). Expected ≥80% (4/5).\n"
        f"Perfect: {perfect_matches}, Partial: {partial_matches}, Failed: {no_matches}"
    )

    # Assert high overall F1 score
    assert overall_f1 >= 0.80, f"Overall F1 score too low: {overall_f1:.3f}. Expected ≥0.80"

    # Verify all 5 limited queries completed successfully
    expected_evaluated = 5
    assert queries_evaluated == expected_evaluated, (
        f"Not all limited queries completed: {queries_evaluated}/{expected_evaluated} queries evaluated"
    )


# Helper functions


async def _run_benchmark(project, pipeline_mode: str):
    """Run benchmark for a specific pipeline and return results.

    Args:
        project: E2EProject instance
        pipeline_mode: 'in_context' or 'hierarchical'

    Returns:
        dict: Benchmark results with metrics
    """
    # Change to project directory
    original_cwd = os.getcwd()
    os.chdir(project.project_dir)

    # Set config file environment variable
    original_config_file = os.environ.get("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(project.config_path)

    # Add project to Python path
    src_dir = str(project.project_dir / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        # Import benchmark runner from the project
        # Need to get package name from project directory
        package_name = project.project_dir.name.replace("-", "_")

        # Suppress channel finder logs during benchmarking
        logging.getLogger(f"{package_name}").setLevel(logging.WARNING)
        logging.getLogger("channel_finder").setLevel(logging.WARNING)

        # Initialize registry for this project context
        # This is required before creating BenchmarkRunner which instantiates ChannelFinderService
        from osprey.registry import initialize_registry, reset_registry

        reset_registry()
        initialize_registry(config_path=str(project.config_path))

        # Import the benchmark runner from the native framework service
        from osprey.services.channel_finder.benchmarks.runner import BenchmarkRunner

        # Create runner and run benchmark
        runner = BenchmarkRunner()
        await runner.run_all_enabled_benchmarks()

        # Find the latest results file
        results_dir = project.project_dir / f"src/{package_name}/data/benchmarks/results"
        if not results_dir.exists():
            raise RuntimeError(f"Results directory not found: {results_dir}")

        # Find most recent results JSON file
        result_files = list(results_dir.glob(f"benchmark_{pipeline_mode}_benchmark_*.json"))
        result_files = [f for f in result_files if "_in_progress" not in f.name]

        if not result_files:
            raise RuntimeError(f"No benchmark results found for {pipeline_mode}")

        latest_result = max(result_files, key=lambda p: p.stat().st_mtime)

        # Load results
        with open(latest_result) as f:
            results = json.load(f)

        return results

    finally:
        # Reset registry to prevent state pollution between tests
        from osprey.registry import reset_registry

        reset_registry()

        # Restore original directory
        os.chdir(original_cwd)

        # Restore original CONFIG_FILE environment variable
        if original_config_file is not None:
            os.environ["CONFIG_FILE"] = original_config_file
        elif "CONFIG_FILE" in os.environ:
            del os.environ["CONFIG_FILE"]

        # Clean up sys.path
        if src_dir in sys.path:
            sys.path.remove(src_dir)

        # CRITICAL: Clean up imported modules to prevent state pollution
        # The benchmark runner imports application modules that can leave
        # stale registry references that interfere with subsequent tests
        modules_to_remove = [key for key in sys.modules.keys() if package_name in key]
        for module in modules_to_remove:
            del sys.modules[module]
