"""MCP Channel Finder Benchmark Tests.

Exercises the real Claude Code sub-agent + MCP tool pipeline for channel
finding across all three pipeline modes (hierarchical, middle_layer,
in_context).  Each pipeline gets 10 curated queries selected from the
existing LangGraph benchmark datasets.

These tests validate that the MCP-based channel finder produces results
comparable to the direct Python API channel finder.
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import (
    SDKWorkflowResult,
    init_project,
    run_sdk_query,
)

# ---------------------------------------------------------------------------
# Query datasets — 10 per pipeline, curated from existing benchmarks
# ---------------------------------------------------------------------------

HIERARCHICAL_QUERIES = [
    {
        "id": "hier_01_dipole_current_sp",
        "user_query": "Set the current in dipole magnet 5",
        "expected": ["MAG:DIPOLE[B05]:CURRENT:SP"],
    },
    {
        "id": "hier_02_rf_tuner",
        "user_query": "What's the RF cavity 2 tuner position?",
        "expected": ["RF:CAVITY[C2]:TUNER:RB"],
    },
    {
        "id": "hier_03_bending_synonym",
        "user_query": "Show me the bending magnet 3 current setpoint",
        "expected": ["MAG:DIPOLE[B03]:CURRENT:SP"],
    },
    {
        "id": "hier_04_range",
        "user_query": "What are the first three dipole current setpoints?",
        "expected": [
            "MAG:DIPOLE[B01]:CURRENT:SP",
            "MAG:DIPOLE[B02]:CURRENT:SP",
            "MAG:DIPOLE[B03]:CURRENT:SP",
        ],
    },
    {
        "id": "hier_05_multi_device",
        "user_query": "What are the horizontal corrector 7 and 9 current setpoints?",
        "expected": [
            "MAG:HCM[H07]:CURRENT:SP",
            "MAG:HCM[H09]:CURRENT:SP",
        ],
    },
    {
        "id": "hier_06_cross_system",
        "user_query": "Is vacuum valve 2 open?",
        "expected": ["VAC:VALVE[V02]:POSITION:OPEN"],
    },
    {
        "id": "hier_07_semantic",
        "user_query": "What's the beam current?",
        "expected": ["DIAG:DCCT[MAIN]:CURRENT:RB"],
    },
    {
        "id": "hier_08_ambiguous",
        "user_query": "What are the RF frequency settings?",
        "expected": [
            "RF:CAVITY[C1]:FREQUENCY:SP",
            "RF:CAVITY[C2]:FREQUENCY:SP",
        ],
    },
    {
        "id": "hier_09_multi_family",
        "user_query": "What are the vacuum levels?",
        "expected": [
            "VAC:ION-PUMP[SR01]:PRESSURE:RB",
            "VAC:ION-PUMP[SR02]:PRESSURE:RB",
            "VAC:ION-PUMP[SR03]:PRESSURE:RB",
            "VAC:ION-PUMP[SR04]:PRESSURE:RB",
            "VAC:ION-PUMP[SR05]:PRESSURE:RB",
            "VAC:ION-PUMP[SR06]:PRESSURE:RB",
            "VAC:GAUGE[SR01A]:PRESSURE:RB",
            "VAC:GAUGE[SR01B]:PRESSURE:RB",
            "VAC:GAUGE[SR02A]:PRESSURE:RB",
            "VAC:GAUGE[SR02B]:PRESSURE:RB",
            "VAC:GAUGE[SR03A]:PRESSURE:RB",
            "VAC:GAUGE[SR03B]:PRESSURE:RB",
        ],
    },
    {
        "id": "hier_10_all_devices",
        "user_query": "Are all dipole magnets ready?",
        "expected": [f"MAG:DIPOLE[B{i:02d}]:STATUS:READY" for i in range(1, 25)],
    },
]

MIDDLE_LAYER_QUERIES = [
    {
        "id": "ml_01_beam_current",
        "user_query": "What is the beam current in the storage ring?",
        "expected": ["SR:DCCT:Current"],
    },
    {
        "id": "ml_02_qf_sector",
        "user_query": "What are the focusing quadrupole currents in sector 1?",
        "expected": ["SR01C:QF1:Current", "SR01C:QF2:Current"],
    },
    {
        "id": "ml_03_rf_setpoints",
        "user_query": "What are the RF voltage setpoints?",
        "expected": ["SR:RF1:VoltSet", "SR:RF2:VoltSet"],
    },
    {
        "id": "ml_04_hcm_readback",
        "user_query": "Show me horizontal corrector current readbacks for sector 2, first two devices",
        "expected": ["SR02C:HCM1:Current", "SR02C:HCM2:Current"],
    },
    {
        "id": "ml_05_ion_pump",
        "user_query": "Get ion pump pressures in sector 1, first two pumps",
        "expected": ["SR01C:VAC:IP1:Pressure", "SR01C:VAC:IP2:Pressure"],
    },
    {
        "id": "ml_06_bts_kicker",
        "user_query": "Show me the injection kicker voltage readback",
        "expected": ["BTS:KICKER:Voltage"],
    },
    {
        "id": "ml_07_br_beam",
        "user_query": "What is the booster ring beam current?",
        "expected": ["BR:DCCT:Current"],
    },
    {
        "id": "ml_08_rf_power",
        "user_query": "Get RF forward power",
        "expected": ["SR:RF1:PowerFwd", "SR:RF2:PowerFwd"],
    },
    {
        "id": "ml_09_corrector_setpoints",
        "user_query": "What are all the corrector magnet setpoints in sector 2, devices 1-2?",
        "expected": [
            "SR02C:HCM1:SetCurrent",
            "SR02C:HCM2:SetCurrent",
            "SR02C:VCM1:SetCurrent",
            "SR02C:VCM2:SetCurrent",
        ],
    },
    {
        "id": "ml_10_cross_system_bpm",
        "user_query": (
            "Show me horizontal BPM positions for first device in sector 1 of SR,"
            " and first device of BR and BTS"
        ),
        "expected": [
            "SR01C:BPM1:X",
            "BR:BPM1:X",
            "BTS:BPM1:X",
        ],
    },
]

IN_CONTEXT_QUERIES = [
    {
        "id": "ic_01_terminal_voltage",
        "user_query": "What's the terminal voltage right now?",
        "expected": ["TerminalVoltageReadBack"],
    },
    {
        "id": "ic_02_filament_rbv",
        "user_query": "Check the electron gun filament current RBV",
        "expected": ["FilamentCurrentRB"],
    },
    {
        "id": "ic_03_dipole_sp",
        "user_query": "What's the setpoint for dipole magnet 5?",
        "expected": ["DipoleMagnet05SetPoint"],
    },
    {
        "id": "ic_04_steering_axis",
        "user_query": "Give me steering coil 7 horizontal setpoint",
        "expected": ["SteeringCoil07XSetPoint"],
    },
    {
        "id": "ic_05_location_axis",
        "user_query": "Show me all the horizontal beam steering setpoints at the accelerating tube",
        "expected": ["SX3Set", "SX40Set"],
    },
    {
        "id": "ic_06_vacuum_synonym",
        "user_query": "What are the vacuum levels in beamline 1?",
        "expected": ["IP41Pressure", "IP78Pressure", "IP125Pressure"],
    },
    {
        "id": "ic_07_pulse_synonym",
        "user_query": "What's the beam pulse width setting?",
        "expected": ["BeamPulseDuration"],
    },
    {
        "id": "ic_08_vert_steering",
        "user_query": "Show me all vertical steering coils in the decelerating tube",
        "expected": ["SY233Set", "SY233RB"],
    },
    {
        "id": "ic_09_multi_axis",
        "user_query": "Show me all steering coil 15 values",
        "expected": [
            "SteeringCoil15XSetPoint",
            "SteeringCoil15XReadBack",
            "SteeringCoil15YSetPoint",
            "SteeringCoil15YReadBack",
        ],
    },
    {
        "id": "ic_10_semantic_group",
        "user_query": "Show me the electron gun diagnostics",
        "expected": [
            "GunPressure",
            "FilamentCurrentSet",
            "FilamentCurrentRB",
            "GunVoltageSet",
            "GunVoltageRB",
            "GridSet",
            "GridRB",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_channels_from_result(result: SDKWorkflowResult) -> list[str]:
    """Extract channel addresses from submit_response tool calls in traces.

    The channel-finder sub-agent calls ``submit_response`` with
    ``data_type="channel_addresses"`` and the resolved PV names in
    ``entry_ids``.  We look for the last such call and return the list.
    """
    candidates = result.tools_matching("submit_response")

    # Prefer calls with data_type="channel_addresses"
    for trace in reversed(candidates):
        inp = trace.input or {}
        if inp.get("data_type") == "channel_addresses":
            ids = inp.get("entry_ids")
            if isinstance(ids, list):
                return ids

    # Fallback: any submit_response with entry_ids
    for trace in reversed(candidates):
        inp = trace.input or {}
        ids = inp.get("entry_ids")
        if isinstance(ids, list) and ids:
            return ids

    return []


def compute_f1(
    predicted: list[str], expected: list[str]
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from predicted and expected channel lists."""
    pred_set = set(predicted)
    exp_set = set(expected)

    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not exp_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_benchmark(
    results: list[dict],
) -> dict:
    """Aggregate benchmark results into summary statistics.

    Args:
        results: List of dicts with keys: query_id, f1, precision, recall, cost_usd.

    Returns:
        Dict with perfect_count, partial_count, no_match_count, overall_f1,
        total_cost, perfect_match_rate.
    """
    perfect = sum(1 for r in results if r["f1"] == 1.0)
    partial = sum(1 for r in results if 0 < r["f1"] < 1.0)
    no_match = sum(1 for r in results if r["f1"] == 0.0)
    overall_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0
    total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)

    return {
        "perfect_count": perfect,
        "partial_count": partial,
        "no_match_count": no_match,
        "overall_f1": overall_f1,
        "total_cost": total_cost,
        "perfect_match_rate": perfect / len(results) if results else 0.0,
        "total_queries": len(results),
    }


# ---------------------------------------------------------------------------
# Results cache — avoids re-running queries in aggregate tests
# ---------------------------------------------------------------------------

# Keyed by query "id"; populated by parametrized tests, consumed by aggregates.
_results_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------


async def _run_single_query(
    project_dir, query_entry: dict, *, use_cache: bool = False
) -> dict:
    """Run a single channel-finder query and return scored result.

    When *use_cache* is True, return a previously cached result if available
    (used by aggregate tests to avoid re-running queries already executed by
    the parametrized tests).
    """
    qid = query_entry["id"]
    if use_cache and qid in _results_cache:
        return _results_cache[qid]

    user_query = query_entry["user_query"]
    expected = query_entry["expected"]

    prompt = f"Find the following channels: {user_query}"

    result = await run_sdk_query(
        project_dir,
        prompt,
        max_turns=30,
        max_budget_usd=0.20,
        model="anthropic/claude-haiku",
    )

    predicted = extract_channels_from_result(result)
    precision, recall, f1 = compute_f1(predicted, expected)

    entry = {
        "query_id": query_entry["id"],
        "user_query": user_query,
        "expected": expected,
        "predicted": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost_usd": result.cost_usd,
        "num_turns": result.num_turns,
    }

    # Diagnostic output
    status = "PERFECT" if f1 == 1.0 else ("PARTIAL" if f1 > 0 else "MISS")
    print(f"\n  [{status}] {query_entry['id']}: F1={f1:.2f} P={precision:.2f} R={recall:.2f}")
    if predicted != expected:
        missing = set(expected) - set(predicted)
        extra = set(predicted) - set(expected)
        if missing:
            print(f"    Missing: {sorted(missing)}")
        if extra:
            print(f"    Extra:   {sorted(extra)}")

    _results_cache[qid] = entry
    return entry


# ---------------------------------------------------------------------------
# Fixtures — module-scoped, one per pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hierarchical_project(tmp_path_factory):
    """Project initialized with hierarchical channel finder mode."""
    tmp = tmp_path_factory.mktemp("bench-hier")
    return init_project(tmp, "bench-hier", channel_finder_mode="hierarchical")


@pytest.fixture(scope="module")
def middle_layer_project(tmp_path_factory):
    """Project initialized with middle_layer channel finder mode."""
    tmp = tmp_path_factory.mktemp("bench-ml")
    return init_project(tmp, "bench-ml", channel_finder_mode="middle_layer")


@pytest.fixture(scope="module")
def in_context_project(tmp_path_factory):
    """Project initialized with in_context channel finder mode."""
    tmp = tmp_path_factory.mktemp("bench-ic")
    return init_project(tmp, "bench-ic", channel_finder_mode="in_context")


# ---------------------------------------------------------------------------
# Test classes — Hierarchical pipeline
# ---------------------------------------------------------------------------


class TestHierarchicalMCPBenchmark:
    """Individual query tests for the hierarchical pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_entry",
        HIERARCHICAL_QUERIES,
        ids=[q["id"] for q in HIERARCHICAL_QUERIES],
    )
    async def test_query(self, hierarchical_project, query_entry):
        entry = await _run_single_query(hierarchical_project, query_entry)
        # Individual queries are informational — no hard assertion per query.
        # Aggregate thresholds are checked in TestHierarchicalMCPAggregate.
        assert entry["f1"] >= 0.0, "F1 must be non-negative"


class TestHierarchicalMCPAggregate:
    """Aggregate threshold checks for the hierarchical pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_overall_thresholds(self, hierarchical_project):
        results = []
        for q in HIERARCHICAL_QUERIES:
            entry = await _run_single_query(hierarchical_project, q, use_cache=True)
            results.append(entry)

        summary = score_benchmark(results)
        print(f"\n  Hierarchical aggregate: {summary}")

        assert summary["perfect_match_rate"] >= 0.60, (
            f"Perfect match rate {summary['perfect_match_rate']:.0%} < 60% threshold"
        )
        assert summary["overall_f1"] >= 0.65, (
            f"Overall F1 {summary['overall_f1']:.2f} < 0.65 threshold"
        )


# ---------------------------------------------------------------------------
# Test classes — Middle Layer pipeline
# ---------------------------------------------------------------------------


class TestMiddleLayerMCPBenchmark:
    """Individual query tests for the middle layer pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_entry",
        MIDDLE_LAYER_QUERIES,
        ids=[q["id"] for q in MIDDLE_LAYER_QUERIES],
    )
    async def test_query(self, middle_layer_project, query_entry):
        entry = await _run_single_query(middle_layer_project, query_entry)
        assert entry["f1"] >= 0.0, "F1 must be non-negative"


class TestMiddleLayerMCPAggregate:
    """Aggregate threshold checks for the middle layer pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_overall_thresholds(self, middle_layer_project):
        results = []
        for q in MIDDLE_LAYER_QUERIES:
            entry = await _run_single_query(middle_layer_project, q, use_cache=True)
            results.append(entry)

        summary = score_benchmark(results)
        print(f"\n  Middle Layer aggregate: {summary}")

        assert summary["perfect_match_rate"] >= 0.60, (
            f"Perfect match rate {summary['perfect_match_rate']:.0%} < 60% threshold"
        )
        assert summary["overall_f1"] >= 0.65, (
            f"Overall F1 {summary['overall_f1']:.2f} < 0.65 threshold"
        )


# ---------------------------------------------------------------------------
# Test classes — In-Context pipeline
# ---------------------------------------------------------------------------


class TestInContextMCPBenchmark:
    """Individual query tests for the in-context pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "query_entry",
        IN_CONTEXT_QUERIES,
        ids=[q["id"] for q in IN_CONTEXT_QUERIES],
    )
    async def test_query(self, in_context_project, query_entry):
        entry = await _run_single_query(in_context_project, query_entry)
        assert entry["f1"] >= 0.0, "F1 must be non-negative"


class TestInContextMCPAggregate:
    """Aggregate threshold checks for the in-context pipeline."""

    @pytest.mark.e2e
    @pytest.mark.e2e_benchmark
    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_overall_thresholds(self, in_context_project):
        results = []
        for q in IN_CONTEXT_QUERIES:
            entry = await _run_single_query(in_context_project, q, use_cache=True)
            results.append(entry)

        summary = score_benchmark(results)
        print(f"\n  In-Context aggregate: {summary}")

        assert summary["perfect_match_rate"] >= 0.60, (
            f"Perfect match rate {summary['perfect_match_rate']:.0%} < 60% threshold"
        )
        assert summary["overall_f1"] >= 0.65, (
            f"Overall F1 {summary['overall_f1']:.2f} < 0.65 threshold"
        )
