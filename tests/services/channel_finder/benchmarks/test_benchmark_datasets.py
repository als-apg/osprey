"""Drift sentinel for the unified cross-paradigm benchmark queries.

Validates that every ``targeted_pv`` in the tier{1,2,3}_queries.json files
shipped under ``data/benchmarks/cross_paradigm/queries/`` resolves against the
matching tier's channel database tree
(``data/channel_databases/tiers/tier{1,2,3}/{in_context,hierarchical,middle_layer}.json``).

This is the cheap, no-LLM cross-paradigm drift sentinel — if the tier DBs
diverge from the queries (as they did pre-2026-05 across paper-vs-core), this
suite fails before any e2e/LLM cost is spent.
"""

from __future__ import annotations

import json

import pytest

from osprey.services.channel_finder.benchmarks.generator import (
    TEMPLATE_DB_PATH,
    validate_queries,
)

# TEMPLATE_DB_PATH lives at <template>/data/channel_databases/hierarchical.json,
# so walk up three components to reach the template root.
TEMPLATE_DIR = TEMPLATE_DB_PATH.parents[2]
QUERIES_DIR = TEMPLATE_DIR / "data" / "benchmarks" / "cross_paradigm" / "queries"
TIERS_ROOT = TEMPLATE_DIR / "data" / "channel_databases" / "tiers"

_TIER_NUMS: tuple[int, ...] = (1, 2, 3)


class TestUnifiedQueriesShipped:
    """All three tier query files must exist alongside the preset."""

    @pytest.mark.parametrize("tier_num", _TIER_NUMS)
    def test_query_file_exists(self, tier_num: int):
        path = QUERIES_DIR / f"tier{tier_num}_queries.json"
        assert path.exists(), f"Missing unified queries file: {path}"

    @pytest.mark.parametrize("tier_num", _TIER_NUMS)
    def test_query_file_is_non_empty_array(self, tier_num: int):
        data = json.loads((QUERIES_DIR / f"tier{tier_num}_queries.json").read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    @pytest.mark.parametrize("tier_num", _TIER_NUMS)
    def test_query_schema(self, tier_num: int):
        data = json.loads((QUERIES_DIR / f"tier{tier_num}_queries.json").read_text())
        for i, entry in enumerate(data):
            assert "user_query" in entry, f"tier{tier_num} entry {i} missing user_query"
            assert "targeted_pv" in entry, f"tier{tier_num} entry {i} missing targeted_pv"
            assert isinstance(entry["targeted_pv"], list), (
                f"tier{tier_num} entry {i} targeted_pv must be a list"
            )
            assert len(entry["targeted_pv"]) > 0, f"tier{tier_num} entry {i} has empty targeted_pv"


class TestUnifiedQueriesValidateAgainstTierDbs:
    """Cross-drift sentinel: every targeted_pv must exist in every paradigm DB
    at that tier.

    Delegates to :func:`validate_queries` (per-tier mode) so the assertion
    semantics here match what ``scripts/generate_benchmark_suite.py --validate``
    enforces at suite-regen time.
    """

    def test_no_drift_across_tiers(self):
        tier_queries = {
            n: QUERIES_DIR / f"tier{n}_queries.json"
            for n in _TIER_NUMS
            if (QUERIES_DIR / f"tier{n}_queries.json").exists()
        }
        assert tier_queries, "No tier query files found — expected tier1/2/3_queries.json"

        result = validate_queries(tier_queries=tier_queries, output_dir=TIERS_ROOT)

        assert not result["missing_databases"], (
            f"Tier DB files missing: {result['missing_databases']}"
        )
        # Format a useful diagnostic on first drift, capped at 10 entries.
        if result["missing"]:
            preview = "\n".join(
                f"  tier{m['tier']}/{m['format']}: {m['pv']}" for m in result["missing"][:10]
            )
            extra = (
                f"\n  ... and {len(result['missing']) - 10} more"
                if len(result["missing"]) > 10
                else ""
            )
            pytest.fail(
                f"{len(result['missing'])} targeted_pv references not found in "
                f"tier DBs:\n{preview}{extra}"
            )
