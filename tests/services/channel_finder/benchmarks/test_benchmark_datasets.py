"""Tests for template benchmark dataset files.

Validates that all three benchmark dataset files (in_context, hierarchical,
middle_layer) exist in the template and reference only valid PVs from their
respective channel databases.
"""

from __future__ import annotations

import json

import pytest

from osprey.services.channel_finder.benchmarks.generator import (
    TEMPLATE_DB_PATH,
    collect_middle_layer_pvs,
    expand_hierarchy,
)

# TEMPLATE_DB_PATH lives at <template>/data/channel_databases/tiers/tier3/hierarchical.json,
# so walk up five components to reach the template root.
TEMPLATE_DIR = TEMPLATE_DB_PATH.parents[4]
DATASETS_DIR = TEMPLATE_DIR / "data" / "benchmarks" / "datasets"
# Validate benchmark PVs against the tier-3 superset (full 4353-channel database).
DATABASES_DIR = TEMPLATE_DIR / "data" / "channel_databases" / "tiers" / "tier3"


class TestBenchmarkDatasetsExist:
    """All three benchmark dataset files must exist in the template."""

    @pytest.mark.parametrize(
        "filename",
        [
            "in_context_benchmark.json",
            "hierarchical_benchmark.json",
            "middle_layer_benchmark.json",
        ],
    )
    def test_benchmark_file_exists(self, filename: str):
        path = DATASETS_DIR / filename
        assert path.exists(), f"Missing benchmark dataset: {path}"

    @pytest.mark.parametrize(
        "filename",
        [
            "in_context_benchmark.json",
            "hierarchical_benchmark.json",
            "middle_layer_benchmark.json",
        ],
    )
    def test_benchmark_is_valid_json_array(self, filename: str):
        data = json.loads((DATASETS_DIR / filename).read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.parametrize(
        "filename",
        [
            "in_context_benchmark.json",
            "hierarchical_benchmark.json",
            "middle_layer_benchmark.json",
        ],
    )
    def test_benchmark_query_schema(self, filename: str):
        data = json.loads((DATASETS_DIR / filename).read_text())
        for i, entry in enumerate(data):
            assert "user_query" in entry, f"Entry {i} missing user_query"
            assert "targeted_pv" in entry, f"Entry {i} missing targeted_pv"
            assert isinstance(entry["targeted_pv"], list), f"Entry {i} targeted_pv must be a list"
            assert len(entry["targeted_pv"]) > 0, f"Entry {i} has empty targeted_pv"


class TestMiddleLayerBenchmarkPVs:
    """Every targeted PV in middle_layer_benchmark.json must exist in the database."""

    @pytest.fixture(scope="class")
    def all_pvs(self) -> set[str]:
        ml_data = json.loads((DATABASES_DIR / "middle_layer.json").read_text())
        return collect_middle_layer_pvs(ml_data)

    @pytest.fixture(scope="class")
    def benchmark(self) -> list[dict]:
        return json.loads((DATASETS_DIR / "middle_layer_benchmark.json").read_text())

    def test_all_pvs_valid(self, benchmark: list[dict], all_pvs: set[str]):
        missing = []
        for entry in benchmark:
            for pv in entry["targeted_pv"]:
                if pv not in all_pvs:
                    missing.append((entry["user_query"][:50], pv))
        assert not missing, f"PVs not found in middle_layer.json: {missing}"

    def test_query_count(self, benchmark: list[dict]):
        assert len(benchmark) >= 30, f"Expected at least 30 queries, got {len(benchmark)}"


class TestHierarchicalBenchmarkPVs:
    """Every targeted PV in hierarchical_benchmark.json must exist in the database."""

    @pytest.fixture(scope="class")
    def all_pvs(self) -> set[str]:
        tree = json.loads(TEMPLATE_DB_PATH.read_text())
        return {ch["pv"] for ch in expand_hierarchy(tree)}

    @pytest.fixture(scope="class")
    def benchmark(self) -> list[dict]:
        return json.loads((DATASETS_DIR / "hierarchical_benchmark.json").read_text())

    def test_all_pvs_valid(self, benchmark: list[dict], all_pvs: set[str]):
        missing = []
        for entry in benchmark:
            for pv in entry["targeted_pv"]:
                if pv not in all_pvs:
                    missing.append((entry["user_query"][:50], pv))
        assert not missing, f"PVs not found in hierarchical.json: {missing}"


class TestInContextBenchmarkPVs:
    """Every targeted PV in in_context_benchmark.json must be a valid alias."""

    @pytest.fixture(scope="class")
    def benchmark(self) -> list[dict]:
        return json.loads((DATASETS_DIR / "in_context_benchmark.json").read_text())

    def test_has_queries(self, benchmark: list[dict]):
        assert len(benchmark) >= 10
