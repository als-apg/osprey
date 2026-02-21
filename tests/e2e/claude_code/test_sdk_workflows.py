"""SDK-based E2E tests for Claude Code + OSPREY MCP integration.

These tests use the Claude Agent SDK (claude_agent_sdk) to spawn Claude Code
as a subprocess and stream typed Python objects for every step, enabling
assertions on individual tool calls, their inputs, outputs, ordering, and cost.

Requires:
- Claude Code CLI installed
- claude_agent_sdk Python package installed
- ANTHROPIC_API_KEY environment variable set

Safety Note - Permission Bypass:
Tests use permission_mode="bypassPermissions" because:
1. Tests run in isolated tmp_path directories with no real codebase
2. Prompts are controlled and only request data retrieval + plotting
3. The project uses mock connectors (no real EPICS hardware)
4. max_budget_usd caps API spend
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import (
    find_png_files,
    init_project,
    run_sdk_query,
)


class TestClaudeCodeSDKIntegration:
    """SDK-based E2E tests with full tool-call observability."""

    # -------------------------------------------------------------------
    # Test 1 — Smoke test: single tool call observability
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_tool_call_observability_smoke(self, tmp_path):
        """Verify that the SDK ToolTrace collection works correctly.

        Uses a simple prompt that triggers a single MCP tool call
        (channel_find) to prove the observability pipeline works.
        """
        project_dir = init_project(tmp_path, "sdk-smoke-test")

        prompt = (
            "Use the channel_find tool to search for BPM channels. "
            "Just report what you find — no plots needed."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=5,
            max_budget_usd=0.25,
        )

        # -- Debug output --
        print("\n--- SDK smoke test ---")
        print(f"  tools called: {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        print(f"  is_error: {result.result.is_error}" if result.result else "  result: N/A")
        for trace in result.tool_traces:
            print(f"  tool: {trace.name}")
            print(f"    input keys: {list(trace.input.keys())}")
            print(f"    is_error: {trace.is_error}")
            result_preview = (trace.result or "")[:200]
            print(f"    result preview: {result_preview}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received from SDK"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # At least one MCP tool was called
        assert len(result.tool_traces) > 0, (
            "No tool calls recorded — SDK observability may be broken"
        )

        # At least one tool name should contain a recognizable MCP tool
        mcp_tool_names = [t.name for t in result.tool_traces if "channel" in t.name.lower()]
        assert len(mcp_tool_names) > 0, (
            f"Expected a channel-related tool call but got: {result.tool_names}"
        )

        # Cost should be reasonable for a simple query
        if result.cost_usd is not None:
            assert result.cost_usd < 0.25, (
                f"Smoke test cost ${result.cost_usd:.4f} — exceeded $0.25 budget"
            )

    # -------------------------------------------------------------------
    # Test 2 — Archiver + plot (hardcoded channels, no channel-finder)
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_archiver_and_plot_via_sdk(self, tmp_path):
        """Verify archiver_read -> execute pipeline with tool ordering.

        Uses hardcoded channel names to bypass the channel-finder and
        reduce LLM non-determinism and cost.
        """
        project_dir = init_project(tmp_path, "sdk-archiver-plot")

        prompt = (
            "Use the archiver_read tool to retrieve data for channels "
            "'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', "
            "'DIAG:BPM03:POSITION:X' over the last 24 hours. "
            "Then use the execute tool to create a timeseries plot of "
            "the data and save it as a PNG file in the current directory."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=15,
            max_budget_usd=1.0,
        )

        # -- Debug output --
        print("\n--- SDK archiver+plot test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            print(f"  tool: {trace.name}")
            print(f"    input keys: {list(trace.input.keys())}")
            print(f"    is_error: {trace.is_error}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # archiver_read was called
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, (
            f"archiver_read not called. Tools used: {result.tool_names}"
        )

        # execute was called
        python_calls = result.tools_matching("execute")
        assert len(python_calls) > 0, (
            f"execute not called. Tools used: {result.tool_names}"
        )

        # archiver_read was called BEFORE execute
        archiver_idx = next(
            i for i, t in enumerate(result.tool_traces) if "archiver_read" in t.name
        )
        python_idx = next(
            i for i, t in enumerate(result.tool_traces) if "execute" in t.name
        )
        assert archiver_idx < python_idx, (
            f"archiver_read (idx={archiver_idx}) should come before "
            f"execute (idx={python_idx})"
        )

        # PNG artifact exists in the project tree
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found in the project -- execute may not have created a plot."
        )

        # Cost should be under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

        print(f"  PNG files: {[p.name for p in png_files]}")

    # -------------------------------------------------------------------
    # Test 3 — Full BPM correlation pipeline (channel-finder -> archiver -> plot)
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_full_bpm_correlation_pipeline_via_sdk(self, tmp_path):
        """Full multi-tool pipeline with sub-agent delegation.

        Natural language prompt matching the user's interactive workflow:
        channel-finder sub-agent -> archiver_read -> execute -> artifact.
        """
        project_dir = init_project(tmp_path, "sdk-bpm-pipeline")

        prompt = (
            "Give me a timeseries and a correlation plot of all horizontal "
            "BPM positions over the last 24 hours. Use the channel_find tool "
            "to discover BPM channels, then archiver_read to get historical "
            "data, then the execute tool to create the plots. Save the plots "
            "as PNG files."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=25,
            max_budget_usd=2.0,
        )

        # -- Debug output --
        print("\n--- SDK full BPM pipeline test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        print(f"  system messages: {len(result.system_messages)}")
        for trace in result.tool_traces:
            error_flag = " [ERROR]" if trace.is_error else ""
            parent_flag = (
                f" (sub-agent: {trace.parent_tool_use_id})"
                if trace.parent_tool_use_id
                else ""
            )
            print(f"  tool: {trace.name}{error_flag}{parent_flag}")
            print(f"    input keys: {list(trace.input.keys())}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # Channel finding was invoked (either via channel_find tool or sub-agent)
        channel_calls = result.tools_matching("channel")
        assert len(channel_calls) > 0, (
            f"No channel-related tool calls found. Tools used: {result.tool_names}"
        )

        # archiver_read was called (should retrieve data for multiple channels)
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, (
            f"archiver_read not called. Tools used: {result.tool_names}"
        )

        # execute was called (should contain plotting code)
        python_calls = result.tools_matching("execute")
        assert len(python_calls) > 0, (
            f"execute not called. Tools used: {result.tool_names}"
        )

        # Check that execute input contains plotting-related code
        plot_related = False
        for call in python_calls:
            code = call.input.get("code", "")
            if any(
                kw in code.lower()
                for kw in ["plot", "figure", "correlation", "plotly", "matplotlib"]
            ):
                plot_related = True
                break
        assert plot_related, (
            "execute was called but code doesn't contain plot-related keywords"
        )

        # At least one PNG artifact was created
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found -- execute may not have created plots."
        )

        # Cost should be under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

        # Turn count should be reasonable
        if result.num_turns is not None:
            assert result.num_turns < 25, (
                f"Test used {result.num_turns} turns — may indicate a loop"
            )

        print(f"  PNG files: {[p.name for p in png_files]}")
        print(f"  archiver calls: {len(archiver_calls)}")
        print(f"  python calls: {len(python_calls)}")

    # -------------------------------------------------------------------
    # Test 4 — 3D scatter plot via data-visualizer subagent
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_3d_scatter_plot_via_data_visualizer(self, tmp_path):
        """Verify the data-visualizer agent can create a 3D scatter plot.

        Regression test: mpl_toolkits.mplot3d must be whitelisted in the
        sandbox executor's import allowlist, otherwise the agent falls back
        to Plotly or fails entirely when asked for a 3D matplotlib plot.

        Pipeline: archiver_read -> data-visualizer subagent -> create_static_plot
        or create_interactive_plot -> artifact with 3D scatter.

        Cost budget: $2.00
        """
        project_dir = init_project(tmp_path, "sdk-3d-scatter")

        prompt = (
            "Use archiver_read to retrieve data for channels "
            "'DIAG:BPM[BPM18]:POSITION:X', 'DIAG:BPM[BPM19]:POSITION:X', "
            "'DIAG:BPM[BPM20]:POSITION:X' over the last 24 hours with "
            "processing 'mean' and bin_size 60. "
            "Then create a 3D scatter plot with BPM18 on X-axis, BPM19 on "
            "Y-axis, BPM20 on Z-axis, colored by time. "
            "Use the data-visualizer agent to create the plot."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=25,
            max_budget_usd=2.0,
        )

        # -- Debug output --
        print("\n--- SDK 3D scatter plot test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            error_flag = " [ERROR]" if trace.is_error else ""
            parent_flag = (
                f" (sub-agent: {trace.parent_tool_use_id})"
                if trace.parent_tool_use_id
                else ""
            )
            print(f"  tool: {trace.name}{error_flag}{parent_flag}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, (
            f"SDK query ended in error: {result.result.result}"
        )

        # archiver_read was called
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, (
            f"archiver_read not called. Tools used: {result.tool_names}"
        )

        # A visualization tool was called (static or interactive)
        viz_calls = (
            result.tools_matching("create_static_plot")
            + result.tools_matching("create_interactive_plot")
        )
        assert len(viz_calls) > 0, (
            f"No visualization tool called. Tools used: {result.tool_names}"
        )

        # The viz tool should have succeeded (no errors)
        viz_errors = [t for t in viz_calls if t.is_error]
        assert len(viz_errors) == 0, (
            f"Visualization tool returned errors: "
            f"{[t.result[:300] for t in viz_errors]}"
        )

        # Check that the viz code contains 3D-related keywords
        has_3d_code = False
        for call in viz_calls:
            code = call.input.get("code", "")
            if any(kw in code.lower() for kw in [
                "3d", "scatter3d", "mplot3d", "axes3d", "projection",
            ]):
                has_3d_code = True
                break
        assert has_3d_code, (
            "Visualization tool was called but code doesn't contain 3D keywords. "
            f"Code snippets: {[c.input.get('code', '')[:200] for c in viz_calls]}"
        )

        # Cost should be under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

        print(f"  viz calls: {len(viz_calls)}")
        print(f"  viz errors: {len(viz_errors)}")
