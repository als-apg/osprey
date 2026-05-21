"""End-to-end test for the RF-cavity-C1 thermal-excursion scenario.

Operator-style prompt — one diagnostic imperative, one deliverable, no
subsystem hints:

    "The beam dumped this morning. Figure out what happened and plot the data."

This is the cross-paradigm benchmark's RF scenario. It tests whether the
agent can, from a bare beam-dump report, (a) enumerate the canonical
suspects for unplanned beam loss (RF trip / vacuum event / magnet fault /
etc.) without being told, (b) discover and retrieve the relevant telemetry
for each suspect, (c) commit to a single cavity (C1 / device 01) as the
fault source based on the telemetry signature, and (d) name the mechanism
(thermal detuning → reflected-power spike → forward-power trip).

A multi-day logbook arc is seeded for cross-source enrichment:
DEMO-026 (trip) → DEMO-027 (investigation identifying cooling-manifold
blockage) → DEMO-028 (manifold flush repair). The logbook seed dates are
re-anchored at ``osprey build`` time so the most recent entry lands two
days before today (see ``rebase_logbook_timestamps`` in
``src/osprey/cli/templates/scaffolding.py``). The archiver pattern is
window-relative: the three C1 excursions appear at normalized positions
(t=0.20, 0.55, 0.85) in any window the agent chooses, so the test is
date-agnostic.

The agent must:

1. Decompose "beam dumped" into the canonical subsystem suspects and
   dispatch parallel investigations (the CLAUDE.md template guides this).
2. Discover RF cavity + DCCT + vacuum-gauge channel addresses via
   channel-finder.
3. Pull a sensible time window via ``mcp__controls__archiver_read`` for
   both cavities (so it can contrast C1 against C2) and at least DCCT.
4. Produce a plot via the data-visualizer subagent showing the correlated
   excursions on C1 with C2 stable for reference.
5. Identify cavity C1 (not C2) as the source and name the thermal-detuning
   mechanism.

Run with:
    pytest tests/e2e/test_rf_cavity_correlation_scenario.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.judge import LLMJudge
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    _default_opus_model,
    init_project,
    run_sdk_query,
)
from tests.e2e.test_preset_agentic import (
    _channel_finder_server_name,
    _to_workflow_result,
)

# We only gate on HAS_SDK — the Claude Agent SDK ships a bundled ``claude``
# binary inside its package, so a system-PATH ``claude --version`` check
# (which is_claude_code_available() runs) would spuriously skip this test
# on CI runners where the SDK is installed but no system ``claude`` is.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    # Passes locally; flaky on CI runners — pending investigation.
    pytest.mark.skipif(
        os.environ.get("GITHUB_ACTIONS") == "true",
        reason="flaky on CI runners; passes locally — pending investigation",
    ),
]


@pytest.mark.order(1)  # actively debugging; run first for fail-fast feedback in CI
@pytest.mark.asyncio
async def test_rf_cavity_c1_correlation_flow(tmp_path: Path) -> None:
    """Operator reports a beam dump; agent must cross-reference logbook +
    telemetry and finger cavity C1 thermal excursions as the cause.

    Tests the full puzzle:
        phenomenon → parallel investigation (logbook + telemetry) →
        channel discovery → archiver retrieval → visualization → root cause.

    Tool-trace assertions are the deterministic contract; the LLM judge
    layer guards against the agent fetching data but failing to name C1.

    Passes locally; flaky on CI runners — skipped on CI pending investigation.
    """
    # Use Opus for the planner: this scenario tests diagnostic reasoning
    # (decompose phenomenon → suspects → cross-correlate → commit to root
    # cause), which Haiku reliably bails on by dumping data and asking the
    # user to interpret it. The data-visualizer / channel-finder subagents
    # still use their per-agent tier defaults from the resolver.
    project = init_project(
        tmp_path,
        "rf_correlation_demo",
        template="control_assistant",
        provider="als-apg",
        model="opus",
    )
    cf_server = _channel_finder_server_name(project)
    if cf_server is None:
        pytest.skip("control-assistant preset has no channel-finder server")

    judge = LLMJudge(provider="als-apg")
    # Operator-style prompt: one diagnostic imperative + one deliverable.
    # No subsystem names, no PV addresses, no cavity ID, no time window.
    query = "The beam dumped this morning. Figure out what happened and plot the data."
    result = await run_sdk_query(
        project,
        query,
        max_turns=50,
        max_budget_usd=30.0,
        model=_default_opus_model(project),
    )

    # --- Tool routing contract -------------------------------------------------
    # Logbook investigation path — the agent must consult the logbook.
    # Tool calls inside the logbook-search subagent surface as
    # mcp__ariel__* in the parent tool trace.
    ariel_calls = [t for t in result.tool_traces if t.name.startswith("mcp__ariel__")]
    assert ariel_calls, (
        f"agent did not call any mcp__ariel__* tool — it did not consult "
        f"the logbook. Tools called: {result.tool_names}"
    )

    # Telemetry investigation path — channel discovery.
    cf_calls = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
    assert cf_calls, (
        f"agent did not call any mcp__channel-finder__* tool — it did not "
        f"discover channel addresses for the relevant subsystems. "
        f"Tools called: {result.tool_names}"
    )

    # Telemetry investigation path — archiver retrieval.
    archiver_calls = [t for t in result.tool_traces if t.name == "mcp__controls__archiver_read"]
    assert archiver_calls, (
        f"agent did not call mcp__controls__archiver_read — it never "
        f"retrieved time-series data. Tools called: {result.tool_names}"
    )

    # The archiver payload(s) must address RF cavity channels — the entire
    # logbook story is about cavity C1, so an agent that fetches only DCCT
    # has failed to follow the trail.
    archiver_payloads = " ".join(str(t.input) for t in archiver_calls).lower()
    assert "cavity" in archiver_payloads or "rf" in archiver_payloads, (
        f"archiver_read called but no RF cavity PV in inputs — agent did "
        f"not retrieve the RF side of the correlation: "
        f"{[t.input for t in archiver_calls]}"
    )

    # Visualization path — the prompt explicitly asks for a plot. The
    # data-visualizer subagent's plot-producing tools live in the
    # osprey_workspace namespace (create_static_plot / create_interactive_plot
    # / create_dashboard), not in a data-visualizer-prefixed namespace.
    plot_tool_names = {
        "mcp__osprey_workspace__create_static_plot",
        "mcp__osprey_workspace__create_interactive_plot",
        "mcp__osprey_workspace__create_dashboard",
    }
    viz_calls = [t for t in result.tool_traces if t.name in plot_tool_names]
    assert viz_calls, (
        f"agent did not produce a plot — no create_static_plot / "
        f"create_interactive_plot / create_dashboard call recorded. "
        f"Tools called: {result.tool_names}"
    )

    # --- Diagnostic conclusion -------------------------------------------------
    # The logbook unambiguously names C1 (DEMO-026/027/028) and the archiver
    # data unambiguously shows three thermal excursions on C1/K1 with C2/K2
    # stable for contrast. The agent must commit to C1 and connect the
    # thermal excursions to the beam dumps.
    eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "Diagnostic-conclusion judging. The tool-trace assertions "
            "above already verify methodology (logbook consulted, "
            "channels discovered, telemetry retrieved, plot produced) — "
            "do not re-penalize those steps.\n"
            "\n"
            "The agent must commit to a specific RF cavity as the fault "
            "source — 'cavity C1', 'cavity 01', 'first cavity', or "
            "equivalent (the logbook uses 'C1' narratively, the channel "
            "database uses device '01' — same instrument, either form "
            "is correct). A non-specific 'an RF cavity', 'one of the "
            "cavities', or 'the RF system' is NOT a pass. The agent "
            "should also offer some causal mechanism — thermal / "
            "cooling / detuning / interlock-related — exact "
            "terminology not required; 'C1 tripped due to a thermal "
            "or cooling issue' is sufficient.\n"
            "\n"
            "Bonus, not required for pass: explicit C2 / device 02 "
            "stable-for-contrast statement, and cross-referencing the "
            "logbook's prior-incident narrative about recurring C1 "
            "thermal events and the cooling-manifold investigation/"
            "repair.\n"
            "\n"
            "Failures: naming C2 instead of C1, attributing the dump "
            "to vacuum / magnets / injection (the archiver data "
            "unambiguously points to RF), or hedging without "
            "committing to a specific cavity."
        ),
    )
    assert eval.passed, eval.reasoning
