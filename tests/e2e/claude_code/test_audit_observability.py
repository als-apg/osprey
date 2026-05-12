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
    find_png_files,
    init_project,
    read_audit_events,
    run_sdk_query,
)


class TestAuditObservability:
    """Natural workflow + transcript-based audit verification."""

    @pytest.mark.requires_api
    @pytest.mark.requires_als_apg
    @pytest.mark.asyncio
    async def test_channel_read_archiver_plot_with_audit(self, tmp_path):
        """Full pipeline: channel_find → archiver_read → execute (plot) + audit.

        Prompts Claude to find horizontal BPM channels, read archiver data for
        the first one, and create a timeseries plot saved as PNG. Then verifies
        that session_log can read OSPREY events from Claude Code transcripts.

        Cost budget: $2.00
        """
        project_dir = init_project(
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
            "3. Use execute to create a timeseries plot and save it "
            "as a PNG file\n"
            "Do not stop until you have completed all three steps and saved "
            "the PNG plot."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=25,
            max_budget_usd=2.00,
        )

        # -- Debug output --
        print("\n--- audit + workflow test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")

        # ================================================================
        # TIER 1: Functional assertions (the workflow worked)
        # ================================================================

        assert result.result is not None, "No ResultMessage received from SDK"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # channel_find (or any channel-finder MCP server tool, in case the
        # agent delegated to the channel-finder subagent which browsed first)
        # must have been called. We require a tool trace, not text narration —
        # the agent claiming "I'll look at BPMs" doesn't satisfy the workflow.
        cf_tools = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
        assert cf_tools, (
            f"No mcp__channel-finder__* tool was called. Tools: {result.tool_names}"
        )

        # archiver_read was called
        archiver_tools = result.tools_matching("archiver_read")
        assert len(archiver_tools) >= 1, f"Expected archiver_read call but got: {result.tool_names}"

        # A plot was created — via execute or create_static_plot/create_interactive_plot
        plot_tool_names = ["execute", "create_static_plot", "create_interactive_plot"]
        plot_tools = [t for t in result.tool_traces if any(p in t.name for p in plot_tool_names)]
        assert len(plot_tools) >= 1, (
            f"Expected a plot tool call (execute or create_*_plot) but got: {result.tool_names}"
        )

        # If execute was used, verify it contains plot-related code
        py_tools = result.tools_matching("execute")
        if py_tools:
            py_code_combined = " ".join(str(t.input.get("code", "")) for t in py_tools).lower()
            plot_keywords = ["plot", "plt", "matplotlib", "savefig", "figure"]
            has_plot_code = any(kw in py_code_combined for kw in plot_keywords)
            assert has_plot_code, (
                f"execute code doesn't appear to create a plot. Code: {py_code_combined[:500]}"
            )

        # At least one PNG artifact exists (or artifact_save was called)
        png_files = find_png_files(project_dir)
        artifact_saves = result.tools_matching("artifact_save")
        assert len(png_files) >= 1 or len(artifact_saves) >= 1, (
            "No PNG files found and no artifact_save calls. The plot may not have been persisted."
        )

        # Tool ordering: channel before archiver, archiver before final plot tool
        # The LLM may retry or reorder intermediate calls, so we check that
        # *some* archiver call precedes the *last* plot tool call.
        tool_names = result.tool_names
        channel_indices = [
            i for i, n in enumerate(tool_names) if n.startswith("mcp__channel-finder__")
        ]
        archiver_indices = [i for i, n in enumerate(tool_names) if "archiver" in n]
        plot_indices = [i for i, n in enumerate(tool_names) if any(p in n for p in plot_tool_names)]
        if channel_indices and archiver_indices and plot_indices:
            assert min(channel_indices) < max(archiver_indices), (
                f"Expected a channel-finder call before an archiver call. "
                f"Channel indices: {channel_indices}, archiver: {archiver_indices}. "
                f"Full order: {tool_names}"
            )
            assert min(archiver_indices) < max(plot_indices), (
                f"Expected an archiver call before the final plot tool. "
                f"Archiver indices: {archiver_indices}, plot: {plot_indices}. "
                f"Full order: {tool_names}"
            )

        # ================================================================
        # TIER 2: Audit observability (transcript-based)
        # ================================================================
        events = read_audit_events(project_dir)

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
            assert result.cost_usd < 2.00, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )
