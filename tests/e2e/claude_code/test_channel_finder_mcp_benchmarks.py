"""MCP Channel Finder Benchmark Tests.

Exercises the real Claude Code sub-agent + MCP tool pipeline for channel
finding across all three pipeline modes (hierarchical, middle_layer,
in_context).  Each pipeline gets 10 curated queries selected from the
existing LangGraph benchmark datasets.

These tests validate that the MCP-based channel finder produces results
comparable to the direct Python API channel finder.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

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
        "expected": ["SR:MAG:DIPOLE:B05:CURRENT:SP"],
    },
    {
        "id": "hier_02_rf_tuner",
        "user_query": "What's the RF cavity 2 tuner position?",
        "expected": ["SR:RF:CAVITY:C2:TUNER:RB"],
    },
    {
        "id": "hier_03_bending_synonym",
        "user_query": "Show me the bending magnet 3 current setpoint",
        "expected": ["SR:MAG:DIPOLE:B03:CURRENT:SP"],
    },
    {
        "id": "hier_04_range",
        "user_query": "What are the first three dipole current setpoints?",
        "expected": [
            "SR:MAG:DIPOLE:B01:CURRENT:SP",
            "SR:MAG:DIPOLE:B02:CURRENT:SP",
            "SR:MAG:DIPOLE:B03:CURRENT:SP",
        ],
    },
    {
        "id": "hier_05_multi_device",
        "user_query": "What are the horizontal corrector 7 and 9 current setpoints?",
        "expected": [
            "SR:MAG:HCM:H07:CURRENT:SP",
            "SR:MAG:HCM:H09:CURRENT:SP",
        ],
    },
    {
        "id": "hier_06_cross_system",
        "user_query": "Is vacuum valve 2 open?",
        "expected": ["SR:VAC:VALVE:V02:POSITION:OPEN"],
    },
    {
        "id": "hier_07_semantic",
        "user_query": "What's the beam current?",
        "expected": ["SR:DIAG:DCCT:MAIN:CURRENT:RB"],
    },
    {
        "id": "hier_08_ambiguous",
        "user_query": "What are the RF frequency settings?",
        "expected": [
            "SR:RF:CAVITY:C1:FREQUENCY:SP",
            "SR:RF:CAVITY:C2:FREQUENCY:SP",
        ],
    },
    {
        "id": "hier_09_multi_family",
        "user_query": "What are the vacuum levels?",
        "expected": [
            "SR:VAC:ION-PUMP:SR01:PRESSURE:RB",
            "SR:VAC:ION-PUMP:SR02:PRESSURE:RB",
            "SR:VAC:ION-PUMP:SR03:PRESSURE:RB",
            "SR:VAC:ION-PUMP:SR04:PRESSURE:RB",
            "SR:VAC:ION-PUMP:SR05:PRESSURE:RB",
            "SR:VAC:ION-PUMP:SR06:PRESSURE:RB",
            "SR:VAC:GAUGE:SR01A:PRESSURE:RB",
            "SR:VAC:GAUGE:SR01B:PRESSURE:RB",
            "SR:VAC:GAUGE:SR02A:PRESSURE:RB",
            "SR:VAC:GAUGE:SR02B:PRESSURE:RB",
            "SR:VAC:GAUGE:SR03A:PRESSURE:RB",
            "SR:VAC:GAUGE:SR03B:PRESSURE:RB",
        ],
    },
    {
        "id": "hier_10_all_devices",
        "user_query": "Are all dipole magnets ready?",
        "expected": [f"SR:MAG:DIPOLE:B{i:02d}:STATUS:READY" for i in range(1, 25)],
    },
]

MIDDLE_LAYER_QUERIES = [
    {
        "id": "ml_01_beam_current",
        "user_query": "What is the beam current in the storage ring?",
        "expected": ["SR:DIAG:DCCT:01:CURRENT:RB"],
    },
    {
        "id": "ml_02_qf_sector",
        "user_query": "What are the focusing quadrupole currents in sector 1?",
        "expected": ["SR:MAG:QF:01:CURRENT:RB", "SR:MAG:QF:02:CURRENT:RB"],
    },
    {
        "id": "ml_03_rf_setpoints",
        "user_query": "What are the RF voltage setpoints?",
        "expected": ["SR:RF:CAVITY:01:VOLTAGE:SP", "SR:RF:CAVITY:02:VOLTAGE:SP"],
    },
    {
        "id": "ml_04_hcm_readback",
        "user_query": "Show me horizontal corrector current readbacks for sector 2, first two devices",
        "expected": ["SR:MAG:HCM:07:CURRENT:RB", "SR:MAG:HCM:08:CURRENT:RB"],
    },
    {
        "id": "ml_05_ion_pump",
        "user_query": "Get ion pump pressures in sector 1, first two pumps",
        "expected": ["SR:VAC:ION-PUMP:01:PRESSURE:RB", "SR:VAC:ION-PUMP:02:PRESSURE:RB"],
    },
    {
        "id": "ml_06_dipole_readback",
        "user_query": "Show me the first dipole current readback",
        "expected": ["SR:MAG:DIPOLE:01:CURRENT:RB"],
    },
    {
        "id": "ml_07_br_beam",
        "user_query": "What is the booster ring beam current?",
        "expected": ["BR:DIAG:DCCT:01:CURRENT:RB"],
    },
    {
        "id": "ml_08_rf_power",
        "user_query": "Get RF forward power",
        "expected": ["SR:RF:CAVITY:01:POWER:FWD", "SR:RF:CAVITY:02:POWER:FWD"],
    },
    {
        "id": "ml_09_corrector_setpoints",
        "user_query": "What are the corrector magnet setpoints for devices 1 and 2?",
        "expected": [
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:HCM:02:CURRENT:SP",
            "SR:MAG:VCM:01:CURRENT:SP",
            "SR:MAG:VCM:02:CURRENT:SP",
        ],
    },
    {
        "id": "ml_10_cross_system_bpm",
        "user_query": ("Show me horizontal BPM positions for the first BPM in both SR and BR"),
        "expected": [
            "SR:DIAG:BPM:01:POSITION:X",
            "BR:DIAG:BPM:01:POSITION:X",
        ],
    },
]

# InContext expected entries are alias groups: each inner list enumerates
# every acceptable surface form for one logical channel (descriptive name
# from in_context.json AND its rendered control-system address). The agent
# scores a hit as long as it surfaces any one of them — InContext's whole
# point is that the descriptive name is a usable affordance, but the address
# is the canonical underlying identifier and equally correct.
IN_CONTEXT_QUERIES = [
    {
        "id": "ic_01_beam_current",
        "user_query": "What's the beam current right now?",
        "expected": [
            ["StorageRing_BeamCurrent_ReadBack", "SR:DIAG:DCCT:MAIN:CURRENT:RB"],
        ],
    },
    {
        "id": "ic_02_beam_lifetime",
        "user_query": "Check the beam lifetime",
        "expected": [
            ["StorageRing_BeamLifetime_ReadBack", "SR:DIAG:DCCT:MAIN:LIFETIME:RB"],
        ],
    },
    {
        "id": "ic_03_dipole_sp",
        "user_query": "What's the setpoint for dipole magnet 5?",
        "expected": [
            ["DipoleMagnet05CurrentSetPoint", "SR:MAG:DIPOLE:B05:CURRENT:SP"],
        ],
    },
    {
        "id": "ic_04_corrector_axis",
        "user_query": "Give me horizontal corrector 7 setpoint",
        "expected": [
            ["HorizontalCorrectorMagnet07CurrentSetPoint", "SR:MAG:HCM:H07:CURRENT:SP"],
        ],
    },
    {
        "id": "ic_05_bpm_position",
        "user_query": "What's beam position at BPM 5?",
        "expected": [
            ["BPM_Position05X", "SR:DIAG:BPM:BPM05:POSITION:X"],
            ["BPM_Position05Y", "SR:DIAG:BPM:BPM05:POSITION:Y"],
        ],
    },
    {
        "id": "ic_06_vacuum_synonym",
        "user_query": "What are the vacuum levels in the storage ring?",
        "expected": [
            ["VacuumIonPump01_Pressure_ReadBack", "SR:VAC:ION-PUMP:SR01:PRESSURE:RB"],
            ["VacuumIonPump02_Pressure_ReadBack", "SR:VAC:ION-PUMP:SR02:PRESSURE:RB"],
            ["VacuumIonPump03_Pressure_ReadBack", "SR:VAC:ION-PUMP:SR03:PRESSURE:RB"],
        ],
    },
    {
        "id": "ic_07_rf_voltage",
        "user_query": "What's the RF cavity 2 voltage?",
        "expected": [
            ["RF_Cavity2_Voltage_ReadBack", "SR:RF:CAVITY:C2:VOLTAGE:RB"],
        ],
    },
    {
        "id": "ic_08_cavity_temps",
        "user_query": "Show me the cavity temperatures",
        "expected": [
            ["RF_Cavity1_Temperature_ReadBack", "SR:RF:CAVITY:C1:TEMPERATURE:RB"],
            ["RF_Cavity2_Temperature_ReadBack", "SR:RF:CAVITY:C2:TEMPERATURE:RB"],
        ],
    },
    {
        "id": "ic_09_quad_readback",
        "user_query": "Give me quadrupole magnet 3 readback",
        "expected": [
            ["FocusingQuad03CurrentReadBack", "SR:MAG:QF:QF03:CURRENT:RB"],
        ],
    },
    {
        "id": "ic_10_dcct_valid",
        "user_query": "Is the DCCT measurement valid?",
        "expected": [
            ["StorageRing_DCCT_Valid", "SR:DIAG:DCCT:MAIN:STATUS:VALID"],
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_response_text(result: SDKWorkflowResult) -> str:
    """Combine all text blocks and tool results into a single string.

    Unlike ``combined_text()`` in sdk_helpers, this preserves original case
    so that channel names can be matched accurately.
    """
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return "\n".join(parts)


# Expected entries are either a plain channel string or an "alias group"
# (list of equivalent surface forms — descriptive name + address). Hierarchical
# and middle-layer queries use the plain form; in-context queries use alias
# groups so a hit on either the descriptive name or the address counts.
ExpectedEntry = str | list[str]


def _aliases(entry: ExpectedEntry) -> list[str]:
    """Normalize an expected entry to its list of acceptable surface forms."""
    return [entry] if isinstance(entry, str) else list(entry)


def _canonical(entry: ExpectedEntry) -> str:
    """Return the canonical (first) surface form for display and set arithmetic."""
    return entry if isinstance(entry, str) else entry[0]


def programmatic_recall_check(
    text: str, expected: list[ExpectedEntry]
) -> tuple[list[str], list[str]]:
    """Check which expected channels appear in the response text.

    Case-insensitive substring match. For alias-group entries, the channel
    is "found" if ANY alias appears in the text. Returns canonical forms
    (the first alias, or the plain string) so callers can present and
    set-compare consistently.

    Returns:
        (found, missing) — canonical channel forms.
    """
    text_lower = text.lower()
    found: list[str] = []
    missing: list[str] = []
    for entry in expected:
        if any(alias.lower() in text_lower for alias in _aliases(entry)):
            found.append(_canonical(entry))
        else:
            missing.append(_canonical(entry))
    return found, missing


class ChannelExtractionResult(BaseModel):
    """Structured output for LLM channel extraction."""

    recommended_channels: list[str]
    reasoning: str


def llm_extract_channels(response_text: str, expected: list[ExpectedEntry]) -> list[str]:
    """Use an LLM judge to extract the agent's final recommended channels.

    Calls Haiku via OSPREY's LiteLLM adapter with structured output to
    distinguish channels in the final answer from those mentioned during
    exploration.
    """
    from osprey.models.providers.litellm_adapter import execute_litellm_completion

    # Flatten alias groups for the judge — it just needs reference samples
    # of what valid channel names look like, not the group structure.
    expected_flat = [alias for entry in expected for alias in _aliases(entry)]
    expected_json = json.dumps(expected_flat, indent=2)
    prompt = (
        "You are evaluating a control system channel finder agent's response.\n"
        "Extract the list of channels that the agent presents as its FINAL\n"
        "recommendation/answer.\n\n"
        "Important distinctions:\n"
        "- Channels mentioned only during exploration or reasoning do NOT count\n"
        "- If the agent lists a broad set then narrows down, only the final "
        "narrowed set counts\n"
        "- Channel names look like PV addresses "
        "(e.g., SR:MAG:DIPOLE:B05:CURRENT:SP)\n"
        "  or descriptive names (e.g., StorageRing_BeamCurrent_ReadBack)\n\n"
        f"Expected channels (for reference — use these to calibrate what "
        f"channel names look like):\n{expected_json}\n\n"
        f"Agent's full response:\n{response_text}"
    )

    # Pick whichever Anthropic-compatible endpoint has a key configured.
    # ALS_APG_API_KEY is the CI-default (AWS Bedrock proxy reachable from
    # GitHub Actions). ANTHROPIC_API_KEY direct is the developer fallback
    # for local runs without ALS group access.
    if os.environ.get("ALS_APG_API_KEY"):
        provider = "als-apg"
        api_key = os.environ["ALS_APG_API_KEY"]
        base_url = "https://llm.gianlucamartino.com"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        api_key = os.environ["ANTHROPIC_API_KEY"]
        base_url = None
    else:
        raise RuntimeError("llm_extract_channels needs ALS_APG_API_KEY or ANTHROPIC_API_KEY")

    result = execute_litellm_completion(
        provider=provider,
        message=prompt,
        model_id="claude-haiku-4-5-20251001",
        api_key=api_key,
        base_url=base_url,
        max_tokens=1024,
        temperature=0.0,
        output_format=ChannelExtractionResult,
    )

    if isinstance(result, ChannelExtractionResult):
        return result.recommended_channels
    # Fallback: if structured output didn't parse, return empty
    return []


def evaluate_channel_response(
    result: SDKWorkflowResult, expected: list[ExpectedEntry]
) -> tuple[list[str], dict]:
    """Two-stage evaluation of a channel finder response.

    Stage 1: Programmatic recall check — are all expected channels present
    in the response text?  If not, recall < 1.0; skip the LLM judge.

    Stage 2: LLM precision judge — extract the agent's FINAL recommended
    channel list from the markdown response for accurate precision scoring.

    Returns:
        (predicted_channels, metadata_dict)
    """
    text = get_response_text(result)
    found, missing = programmatic_recall_check(text, expected)

    meta: dict = {"stage": 1, "found": found, "missing": missing}

    if missing:
        # Stage 1 failure: some expected channels not in text at all.
        # predicted = found channels (assume those are correct → precision=1.0)
        meta["evaluation"] = "programmatic_recall_fail"
        return found, meta

    # Stage 2: all expected channels appear in text — use LLM judge
    # to extract the agent's final recommended list for precision scoring.
    meta["stage"] = 2
    try:
        predicted = llm_extract_channels(text, expected)
        meta["evaluation"] = "llm_judge"
        meta["llm_extracted"] = predicted
    except Exception as exc:
        # If LLM judge fails, fall back to found channels
        meta["evaluation"] = "llm_judge_error"
        meta["llm_error"] = str(exc)
        predicted = found

    return predicted, meta


def compute_f1(predicted: list[str], expected: list[ExpectedEntry]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from predicted channels vs. expected groups.

    A predicted channel matches an expected group if it equals any alias in
    that group (case-insensitive). Each expected group can be matched at most
    once — two predicted aliases for the same logical channel count as one TP
    on the recall side and as one matched item on the precision side, with
    any extras counted against precision.
    """
    pred_set = {p.lower() for p in predicted}
    if not pred_set and not expected:
        return 1.0, 1.0, 1.0
    if not pred_set or not expected:
        return 0.0, 0.0, 0.0

    matched_groups = 0
    matched_preds: set[str] = set()
    for entry in expected:
        aliases_lower = {a.lower() for a in _aliases(entry)}
        hits = pred_set & aliases_lower
        if hits:
            matched_groups += 1
            matched_preds |= hits

    precision = len(matched_preds) / len(pred_set)
    recall = matched_groups / len(expected)
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


@pytest.fixture(scope="session", autouse=True)
def _dump_benchmark_results():
    """Persist per-query results + per-pipeline aggregates as JSON at session
    end so the run produces a real artifact, not just stdout. Path is
    overridable via OSPREY_BENCHMARK_RESULTS_PATH; default is
    tests/e2e/claude_code/benchmark_results/<UTC-timestamp>.json.
    """
    yield
    if not _results_cache:
        return
    import datetime
    from pathlib import Path

    pipelines = {
        "hierarchical": [q["id"] for q in HIERARCHICAL_QUERIES],
        "middle_layer": [q["id"] for q in MIDDLE_LAYER_QUERIES],
        "in_context": [q["id"] for q in IN_CONTEXT_QUERIES],
    }
    aggregates = {
        name: score_benchmark([_results_cache[qid] for qid in ids if qid in _results_cache])
        for name, ids in pipelines.items()
    }
    payload = {
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "aggregates": aggregates,
        "results": _results_cache,
    }

    override = os.environ.get("OSPREY_BENCHMARK_RESULTS_PATH")
    if override:
        out_path = Path(override)
    else:
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path(__file__).parent / "benchmark_results" / f"{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Benchmark results written to: {out_path}")


# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------


async def _run_single_query(project_dir, query_entry: dict, *, use_cache: bool = False) -> dict:
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

    # Don't override the model — let run_sdk_query resolve the project's
    # configured haiku tier (als-apg/haiku in CI, cborg/anthropic/claude-haiku
    # locally if cborg is configured). The earlier hardcoded
    # "anthropic/claude-haiku" string is CBORG-namespace and 404s under
    # als-apg, which was a Cluster A blocker until 2026-04.
    result = await run_sdk_query(
        project_dir,
        prompt,
        max_turns=30,
        max_budget_usd=0.20,
    )

    predicted, eval_meta = evaluate_channel_response(result, expected)
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
        "eval_meta": eval_meta,
    }
    # Attach the full agent response on non-perfect queries so the JSON
    # artifact is self-sufficient for diagnosing MISSes / PARTIALs without
    # re-running. Perfect queries skip this to keep the artifact lean.
    if f1 < 1.0:
        entry["response_text"] = get_response_text(result)

    # Diagnostic output. With alias groups, "missing" is per-group (canonical
    # form) and "extra" is any predicted channel that didn't land in any group.
    status = "PERFECT" if f1 == 1.0 else ("PARTIAL" if f1 > 0 else "MISS")
    print(f"\n  [{status}] {query_entry['id']}: F1={f1:.2f} P={precision:.2f} R={recall:.2f}")
    pred_lower = {p.lower() for p in predicted}
    missing_groups = [
        _canonical(e) for e in expected if not (pred_lower & {a.lower() for a in _aliases(e)})
    ]
    all_aliases_lower = {a.lower() for e in expected for a in _aliases(e)}
    extra = [p for p in predicted if p.lower() not in all_aliases_lower]
    if missing_groups:
        print(f"    Missing: {sorted(missing_groups)}")
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
