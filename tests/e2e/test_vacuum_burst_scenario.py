"""End-to-end test for the Sector-7 vacuum-burst + beam-loss scenario.

This is the talk-demo scenario, set up as the seed for the OSPREY benchmark
suite. The operator prompt is intentionally bare:

    "We lost about 5 mA of beam yesterday around 14:32. Did the vacuum
    do anything weird around then?"

That is the full prompt. No channel names, no PV patterns, no time-window
shape, no "if any" hedging. The agent must do the discovery work:

  1. Translate "vacuum" → pressure gauges and use the channel-finder
     pipeline to resolve actual addresses (``SR:VAC:GAUGE:SR{01..12}:...``).
     Refetching DCCT is welcome but not required: the operator already
     announced the beam-loss event with a timestamp.
  2. Resolve "yesterday around 14:32" to a concrete past timestamp and
     choose a sensible time window straddling 14:32:08, then call
     ``mcp__controls__archiver_read`` to fetch the vacuum time series.
  3. Inspect across sectors and finger Sector 7 (SR07) — the only
     sector whose pressure spike aligns with the beam-loss event
     (ground truth: Pearson r ≈ -0.88 against DCCT; max|r| for the
     other 11 sectors ≤ 0.077).

The ground truth now lives in the preset's simulation overlay,
``data/simulation/machine.json`` (the ``vacuum-burst`` scenario): the SR07
pressure spike + coincident DCCT dip are declared as an ``at_time`` event
anchored daily at 14:32:08. The test activates that scenario after building
the project via ``activate_scenario(project, "vacuum-burst")``; the mock
connectors then route gauge/DCCT reads through the engine instead of the
flat ``nominal`` default.

If the agent can't do this, the failure is in the OSPREY preset config
(channel-finder agent context, archiver tool description, planner prompt) —
fix the config, don't weaken the prompt. This test is the contract.

The afternoon anchor (14:32) is deliberate: it temporally separates this
telemetry-only scenario from the RF-cavity scenario's morning logbook arc
(DEMO-026/027/028, ~2-3 days back), so logbook search cannot contaminate
the vacuum investigation. "Yesterday" rather than "today" keeps the
timestamp unambiguously in the past regardless of when the test runs.

The ``at_time`` event fires for every calendar date whose 14:32:08 falls
inside the requested window, so the test is date-agnostic.

Run with:
    pytest tests/e2e/test_vacuum_burst_scenario.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.judge import LLMJudge
from tests.e2e.sdk_helpers import (
    HAS_SDK,
    _default_opus_model,
    activate_scenario,
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
    # Skipped on CI: needs a pre-seeded ARIEL logbook postgres, which the
    # default GitHub Actions runner does not provision (see tests/e2e/README.md).
    pytest.mark.skipif(
        os.environ.get("GITHUB_ACTIONS") == "true",
        reason="needs local ARIEL logbook backend; not provisioned on CI runners",
    ),
]


@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb rare LLM stochastic misses
@pytest.mark.asyncio
async def test_sector7_vacuum_burst_flow(tmp_path: Path) -> None:
    """Operator gives a vague beam-loss complaint; agent must finger SR07.

    Tests the full intellectual chain:
        phenomenon → subsystems → channel-finder discovery →
        archiver retrieval → cross-channel correlation → root cause.

    Tool-trace assertions are the deterministic contract; the LLM judge
    layer guards against the agent fetching data but failing to identify
    the anomalous sector.

    Runs locally against a seeded ARIEL backend; skipped on CI (no backend).
    Marked ``flaky(reruns=2)`` to absorb the rare stochastic miss where the
    agent retrieves the data but bails before committing to the sector.
    """
    # Use Opus for the planner: this scenario tests diagnostic reasoning
    # (subsystem decomposition → discovery → cross-channel correlation →
    # localized root cause), which Haiku has been observed to bail on by
    # spinning in channel-finder discovery loops without progressing to
    # archiver retrieval. The data-visualizer / channel-finder subagents
    # still use their per-agent tier defaults from the resolver.
    #
    # Tier 3 (full channel DB): the vacuum gauges live only in tier 2+, so a
    # tier-1 build leaves SR:VAC:GAUGE:* undiscoverable and the agent spins in
    # channel-finder without ever retrieving data. Tier 3 (not the merely
    # sufficient tier 2) is used so the discoverable channel set matches the
    # full facility the simulation machine model defines, consistent with the
    # sibling RF scenario.
    project = init_project(
        tmp_path,
        "vacuum_burst_demo",
        template="control_assistant",
        provider="als-apg",
        model="opus",
        tier=3,
    )
    # Switch the mock connectors' data substrate from the flat ``nominal``
    # default to the ``vacuum-burst`` scenario declared in
    # ``data/simulation/machine.json`` — the SR07 pressure spike + DCCT dip
    # anchored daily at 14:32:08 via an ``at_time`` event.
    activate_scenario(project, "vacuum-burst")
    cf_server = _channel_finder_server_name(project)
    if cf_server is None:
        pytest.skip("control-assistant preset has no channel-finder server")

    judge = LLMJudge(provider="als-apg")
    # Operator prompt: exactly what someone on shift would type. No PV
    # names, no subsystem hints, no time-window shape. The whole point of
    # the test is whether the agent can decompose this into the right
    # discovery + retrieval + analysis pipeline.
    query = (
        "We lost about 5 mA of beam yesterday around 14:32. "
        "Did the vacuum do anything weird around then?"
    )
    result = await run_sdk_query(
        project,
        query,
        max_turns=25,
        max_budget_usd=30.0,
        model=_default_opus_model(project),
    )

    # --- Tool routing contract -------------------------------------------------
    # The agent must reach for channel-finder to discover both the
    # beam-current channel and the vacuum gauges.
    cf_calls = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
    assert cf_calls, (
        f"agent did not call any mcp__channel-finder__* tool — it did not "
        f"discover channel addresses on its own. Tools called: {result.tool_names}"
    )

    # The agent must reach for the archiver to fetch historical data.
    archiver_calls = [t for t in result.tool_traces if t.name == "mcp__controls__archiver_read"]
    assert archiver_calls, (
        f"agent did not call mcp__controls__archiver_read — it never "
        f"retrieved time-series data. Tools called: {result.tool_names}"
    )

    # The archiver payload(s) must hit at least one vacuum gauge — that's
    # the literal subject of the operator's question. The DCCT side is NOT
    # required: the operator already announced the beam-loss event with a
    # timestamp, so a competent answer can scope to vacuum at ~14:32
    # without refetching beam current. The LLM judge below is the final
    # arbiter of whether the agent named SR07.
    archiver_payloads = " ".join(str(t.input) for t in archiver_calls).lower()
    assert "gauge" in archiver_payloads or "vac" in archiver_payloads, (
        f"archiver_read called but no vacuum-gauge PV in inputs — agent did "
        f"not retrieve the vacuum side of the correlation: "
        f"{[t.input for t in archiver_calls]}"
    )

    # --- Diagnostic conclusion -------------------------------------------------
    # The data is unambiguous: SR07 is the only sector that correlates
    # with the beam-current dip. If the agent didn't name it, it failed
    # the puzzle even if every tool call was correct.
    eval = await judge.evaluate(
        _to_workflow_result(query, result),
        expectations=(
            "The agent must identify Sector 7 (SR07) as the anomalous "
            "sector whose vacuum pressure spiked at the same instant as the "
            "beam-current dip, and connect that vacuum event to the ~5 mA "
            "beam loss. A passing response names SR07 / Sector 7 "
            "explicitly. Responses that report 'no anomaly found', name a "
            "different sector, or fail to commit to a specific sector are "
            "failures — the underlying data unambiguously points to SR07 "
            "(Pearson r ≈ -0.88 against DCCT; the other 11 sectors stay "
            "flat with max|r| ≤ 0.08)."
        ),
    )
    assert eval.passed, eval.reasoning
