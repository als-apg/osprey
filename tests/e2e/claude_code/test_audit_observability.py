"""E2E test: Natural workflow + Audit observability.

Consolidates the execute functional tests and audit verification into
a single test that exercises the natural user workflow:
  channel_find → archiver_read → execute (plot) → PNG artifact

Then verifies that Claude Code native transcripts contain the expected OSPREY
tool-call events and (optionally) lifecycle events (read via TranscriptReader).

Requires:
- Claude Code CLI installed
- claude_agent_sdk Python package installed
- ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import warnings

import pytest

from tests.e2e.sdk_helpers import (
    e2e_budget_scale,
    find_png_files,
    init_project,
    read_audit_events,
    run_sdk_query_with_hooks,
)

pytestmark = pytest.mark.agentic_benchmark

# The workspace plotting tools are an equivalent capability to running
# matplotlib through the Python executor, and the agent picks between them
# non-deterministically — a prompt asking for Python is a steer, not a
# guarantee. Forbidding them at the SDK level is what makes this test actually
# exercise the Python route it claims to cover; without it the run silently
# proves nothing whenever the agent delegates to the data-visualizer instead.
_WORKSPACE_PLOT_TOOLS = [
    "mcp__osprey_workspace__create_static_plot",
    "mcp__osprey_workspace__create_interactive_plot",
    "mcp__osprey_workspace__create_dashboard",
]


class TestAuditObservability:
    """Natural workflow + transcript-based audit verification."""

    # Multi-step agentic pipeline (channel-finder -> archiver -> Python -> PNG)
    # on Haiku. Reruns absorb ordinary model non-determinism; the strict
    # artifact/audit assertions still gate, and the denied-tool assertion below
    # means a permission regression fails every attempt rather than presenting
    # as flake.
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    @pytest.mark.requires_api
    @pytest.mark.requires_als_apg
    @pytest.mark.asyncio
    async def test_channel_read_archiver_plot_with_audit(self, tmp_path):
        """Full pipeline: channel_find → archiver_read → execute (plot) + audit.

        Prompts Claude to find horizontal BPM channels, read archiver data for
        the first one, and create a timeseries plot saved as PNG. Then verifies
        that session_log can read OSPREY events from Claude Code transcripts.

        Runs through ``run_sdk_query_with_hooks`` rather than the bypass runner:
        ``mcp__python__execute`` ships in ``settings.json`` ``permissions.ask``
        because it is a write-capable tool, and ``bypassPermissions`` does not
        override that static list. With no approver the call comes back refused
        ("you haven't granted it yet"), which reads as a missing plot rather
        than a blocked tool. ``auto_approve`` supplies the operator decision a
        real deployment would, matching test_safety_python.py.

        Cost budget: $2.00
        """
        repo = init_project(
            tmp_path,
            "audit-workflow",
            provider="als-apg",
            channel_finder_mode="in_context",
        )

        prompt = (
            "I need a timeseries plot of horizontal BPM data. Please do ALL "
            "three steps:\n"
            "1. Use channel_find to discover horizontal BPM channels\n"
            "2. Use archiver_read to get the last 1 hour of data for the "
            "first BPM channel\n"
            "3. Use Python to create a timeseries plot and save it "
            "as a PNG file\n"
            "Do not stop until you have completed all three steps and saved "
            "the PNG plot."
        )

        result = await run_sdk_query_with_hooks(
            repo,
            prompt,
            approval_policy="auto_approve",
            max_turns=25,
            max_budget_usd=2.00,
            disallowed_tools=_WORKSPACE_PLOT_TOOLS,
        )

        # -- Debug output --
        print("\n--- audit + workflow test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        print(f"  approval events: {[(e.tool_name, e.decision) for e in result.hook_events]}")
        print(
            f"  errored tools: {[(t.name, str(t.result)[:80]) for t in result.tool_traces if t.is_error]}"
        )

        # ================================================================
        # TIER 1: Functional assertions (the workflow worked)
        # ================================================================

        assert result.result is not None, "No ResultMessage received from SDK"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # No tool call was refused by the permission layer. Scoped to permission
        # refusals rather than any is_error trace, because ordinary tool errors
        # the agent recovers from are legitimate (e.g. Read rejecting an
        # oversized file). Without this, a gated tool returns a refusal, the
        # agent silently produces no artifact, and the failure surfaces as the
        # PNG assertion below — which blames persistence for a permission
        # problem and reads as flake across reruns.
        refusal_markers = ("requested permissions to use", "haven't granted it yet")
        refused = [
            t
            for t in result.tool_traces
            if t.is_error and any(m in str(t.result or "") for m in refusal_markers)
        ]
        assert not refused, (
            "Tool call(s) refused by the permission layer: "
            f"{[(t.name, str(t.result)[:120]) for t in refused]}. "
            "A tool the workflow needs is gated behind approval that this run "
            "cannot answer — check settings.json permissions.ask."
        )

        # channel_find (or any channel-finder MCP server tool, in case the
        # agent delegated to the channel-finder subagent which browsed first)
        # must have been called. We require a tool trace, not text narration —
        # the agent claiming "I'll look at BPMs" doesn't satisfy the workflow.
        cf_tools = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
        assert cf_tools, f"No mcp__channel-finder__* tool was called. Tools: {result.tool_names}"

        # archiver_read was called
        archiver_tools = result.tools_matching("archiver_read")
        assert len(archiver_tools) >= 1, f"Expected archiver_read call but got: {result.tool_names}"

        # The plot came from the Python route specifically. The workspace
        # plotting tools are disallowed for this run, so this is the coverage
        # the test uniquely owns: a PNG artifact produced by agent-authored
        # matplotlib through the executor. The other pipeline tests cover the
        # data-visualizer route and tool-agnostic plotting.
        py_tools = result.tools_matching("execute")
        assert len(py_tools) >= 1, (
            f"Expected an mcp__python__execute call but got: {result.tool_names}"
        )

        py_code_combined = " ".join(str(t.input.get("code", "")) for t in py_tools).lower()
        plot_keywords = ["plot", "plt", "matplotlib", "savefig", "figure"]
        has_plot_code = any(kw in py_code_combined for kw in plot_keywords)
        assert has_plot_code, (
            f"execute code doesn't appear to create a plot. Code: {py_code_combined[:500]}"
        )

        # At least one PNG artifact exists (or artifact_register was called)
        png_files = find_png_files(repo)
        registrations = result.tools_matching("artifact_register")
        assert len(png_files) >= 1 or len(registrations) >= 1, (
            "No PNG files found and no artifact_register calls. The plot may not have been persisted."
        )

        # NOTE: We intentionally do NOT assert chronological tool ordering.
        # Sub-agent tool calls are harvested from the CLI side files *after*
        # the main query() stream and appended to tool_traces (see
        # sdk_helpers._harvest_subagent_traces), so result.tool_names is not a
        # globally time-ordered sequence once the agent delegates — and Haiku
        # routinely runs channel-finding and plotting in sub-agents. An index
        # check like min(channel_indices) < max(archiver_indices) then compares
        # a late-appended sub-agent index against an early main-stream index
        # and fails even when the agent acted in the correct causal order. The
        # "channel_find / archiver_read / plot tool were all invoked"
        # assertions above already capture the workflow; cross-agent ordering
        # is unreconstructable from append order.

        # ================================================================
        # TIER 2: Audit observability (transcript-based)
        # ================================================================
        events = read_audit_events(repo)

        print(f"\n  audit events found: {len(events)}")
        for ev in events:
            print(
                f"    type={ev.get('type')}, tool={ev.get('tool', '-')}, "
                f"server={ev.get('server', '-')}"
            )

        # 1. At least one event was found in the transcript
        assert len(events) >= 1, (
            "No OSPREY tool events found in Claude Code transcripts — "
            "TranscriptReader may not have found a matching session."
        )

        # 2. At least one tool_call event
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        assert len(tool_calls) >= 1, (
            f"No tool_call events. Event types: {[e.get('type') for e in events]}"
        )

        # 3. At least one tool_call from controls or channel-finder server
        control_system_calls = [
            e for e in tool_calls if e.get("server") in ("controls", "channel-finder")
        ]
        assert len(control_system_calls) >= 1, (
            f"No tool_call from control-system or channel-finder server. "
            f"Servers seen: {[e.get('server') for e in tool_calls]}"
        )

        # 4. Each tool_call has required schema fields
        for tc in tool_calls:
            assert "timestamp" in tc, f"tool_call missing timestamp: {tc}"
            assert "tool" in tc, f"tool_call missing tool: {tc}"
            assert "is_error" in tc, f"tool_call missing is_error: {tc}"

        # 5. result_summary is non-empty on at least one event
        has_summary = any(
            tc.get("result_summary") and tc["result_summary"].strip() for tc in tool_calls
        )
        assert has_summary, (
            "No tool_call has a non-empty result_summary — "
            "TranscriptReader may not be extracting tool_result content"
        )

        # ================================================================
        # LIFECYCLE EVENTS (derived from Task tool calls in transcript)
        # ================================================================
        agent_starts = [e for e in events if e.get("type") == "agent_start"]
        agent_stops = [e for e in events if e.get("type") == "agent_stop"]

        print(f"\n  agent_start events: {len(agent_starts)}")
        print(f"  agent_stop events: {len(agent_stops)}")

        if agent_starts:
            for ev in agent_starts:
                assert "agent_type" in ev, f"agent_start missing agent_type: {ev}"
                assert "agent_id" in ev, f"agent_start missing agent_id: {ev}"

            for ev in agent_stops:
                assert "agent_type" in ev, f"agent_stop missing agent_type: {ev}"
                assert "agent_id" in ev, f"agent_stop missing agent_id: {ev}"

            print("  lifecycle events validated successfully")
        else:
            warnings.warn(
                "No agent_start lifecycle events found. "
                "The SDK may not have delegated to a subagent via Task tool. "
                "Tool-call assertions passed — lifecycle tracking is advisory.",
                stacklevel=1,
            )

        # -- Cost guard --
        if result.cost_usd is not None:
            budget = 2.0 * e2e_budget_scale()
            assert result.cost_usd < budget, (
                f"Test cost ${result.cost_usd:.4f} — exceeded ${budget:.2f} budget"
            )
