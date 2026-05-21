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

from tests.e2e.judge import LLMJudge
from tests.e2e.sdk_helpers import (
    find_png_files,
    init_project,
    run_sdk_query,
)
from tests.e2e.test_preset_agentic import _to_workflow_result


class TestClaudeCodeSDKIntegration:
    """SDK-based E2E tests with full tool-call observability."""

    # -------------------------------------------------------------------
    # Test 1 — Smoke test: single tool call observability
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_als_apg
    @pytest.mark.asyncio
    async def test_tool_call_observability_smoke(self, tmp_path):
        """Verify that the SDK ToolTrace collection works correctly.

        Uses a simple prompt that triggers a single MCP tool call
        (channel_find) to prove the observability pipeline works.
        """
        project_dir = init_project(tmp_path, "sdk-smoke-test", provider="als-apg")

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

        # The prompt asks for channel_find specifically — assert a channel-finder
        # MCP server tool was called, not just any "channel"-substring tool
        # (controls-server channel_read/write/limits would otherwise satisfy this).
        cf_tools = [t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")]
        assert cf_tools, f"Expected a channel-finder tool call but got: {result.tool_names}"

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
    @pytest.mark.requires_als_apg
    @pytest.mark.asyncio
    async def test_archiver_and_plot_via_sdk(self, tmp_path):
        """Verify archiver_read → plot pipeline with semantic judging.

        Uses hardcoded channel names to bypass the channel-finder and
        reduce LLM non-determinism and cost.

        Three layers of contract:
          1. Deterministic: ``archiver_read`` was called and a PNG
             artifact exists in the project tree.
          2. Semantic (LLM judge): the agent fetched the requested
             channel data and produced a plot from it — the judge
             decides whether ``execute``, ``create_static_plot``, or
             any other plotting path was used correctly, without the
             test prescribing a specific tool.
          3. Cost: under $1.00 budget.
        """
        project_dir = init_project(tmp_path, "sdk-archiver-plot", provider="als-apg")

        prompt = (
            "Use the archiver_read tool to retrieve data for channels "
            "'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', "
            "'DIAG:BPM03:POSITION:X' over the last 24 hours. "
            "Then create a timeseries plot of the data and save it as a "
            "PNG file."
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

        # -- Deterministic assertions (cheap, catch gross regressions) --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found in the project — agent did not produce a plot."
        )
        print(f"  PNG files: {[p.name for p in png_files]}")

        # -- Semantic assertion via LLM judge --
        # The agent may legitimately solve this via ``execute`` (raw python),
        # ``create_static_plot`` (dedicated visualizer), or another path. The
        # judge evaluates the workflow against the contract, not the tools.
        judge = LLMJudge(provider="als-apg")
        evaluation = await judge.evaluate(
            _to_workflow_result(prompt, result),
            expectations=(
                "The agent must (a) retrieve time-series data for the three "
                "named BPM channels via the archiver, and then (b) plot that "
                "data and save it as a PNG. The retrieval step must occur "
                "before the plotting step. The specific plotting mechanism "
                "(direct python execution vs. a dedicated plotting MCP tool) "
                "is acceptable in any form — do not penalize the choice of "
                "tool. Fail only if data was not actually fetched for the "
                "requested channels, if no plot was produced, or if the "
                "ordering is wrong (e.g., the agent invented data instead of "
                "calling the archiver)."
            ),
        )
        assert evaluation.passed, (
            f"LLM judge failed the workflow.\n"
            f"  reasoning: {evaluation.reasoning}\n"
            f"  confidence: {evaluation.confidence}\n"
            f"  warnings: {evaluation.warnings}\n"
            f"  tools used: {result.tool_names}"
        )

        # -- Cost ceiling --
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

    # -------------------------------------------------------------------
    # Test 3 — Full BPM correlation pipeline (channel-finder -> archiver -> plot)
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_als_apg
    @pytest.mark.asyncio
    async def test_full_bpm_correlation_pipeline_via_sdk(self, tmp_path):
        """Full multi-tool pipeline with sub-agent delegation.

        Natural language prompt matching the user's interactive workflow:
        channel-finder sub-agent → archiver_read → plot → artifact.

        Three layers of contract:
          1. Deterministic: channel-finder invoked, archiver_read called,
             at least one PNG artifact produced, cost + turn ceilings.
          2. Semantic (LLM judge): the agent followed the
             discover-channels → fetch-data → plot pipeline correctly,
             regardless of whether plotting went through ``execute``,
             ``create_static_plot``, or another visualizer tool.
        """
        project_dir = init_project(tmp_path, "sdk-bpm-pipeline", provider="als-apg")

        prompt = (
            "Give me a timeseries and a correlation plot of all horizontal "
            "BPM positions over the last 24 hours. Use the channel_find tool "
            "to discover BPM channels, then archiver_read to get historical "
            "data, then plot the data. Save the plots as PNG files."
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
                f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
            )
            print(f"  tool: {trace.name}{error_flag}{parent_flag}")
            print(f"    input keys: {list(trace.input.keys())}")

        # -- Deterministic assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # Channel finding was invoked. Match the channel-finder MCP server
        # specifically — `tools_matching("channel")` would also accept
        # controls-server `channel_read`/`channel_write`/`channel_limits`,
        # none of which constitute "finding a channel".
        channel_calls = [
            t for t in result.tool_traces if t.name.startswith("mcp__channel-finder__")
        ]
        assert len(channel_calls) > 0, (
            f"No channel-finder tool calls found. Tools used: {result.tool_names}"
        )

        # archiver_read was called (should retrieve data for multiple channels)
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # At least one PNG artifact was created
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, "No PNG files found — agent did not produce plots."
        print(f"  PNG files: {[p.name for p in png_files]}")

        # -- Semantic assertion via LLM judge --
        # The plotting step may go through ``execute`` (raw python),
        # ``create_static_plot`` (dedicated visualizer), or another tool —
        # the judge decides if the discover→fetch→plot pipeline was executed
        # correctly without prescribing the plotting mechanism.
        judge = LLMJudge(provider="als-apg")
        evaluation = await judge.evaluate(
            _to_workflow_result(prompt, result),
            expectations=(
                "The agent must complete a three-step pipeline: (a) discover "
                "horizontal BPM channels via channel-finder, (b) fetch "
                "historical data for those channels via the archiver over a "
                "~24-hour window, (c) produce both a timeseries plot AND a "
                "correlation plot from that data, saved as PNG. Steps must "
                "occur in that order (discover → fetch → plot). The plotting "
                "tool choice (direct python execution vs. a dedicated "
                "plotting MCP tool) is acceptable in any form — do not "
                "penalize the choice. Fail only if a step is missing, the "
                "ordering is wrong, the channels fetched are clearly not "
                "horizontal BPMs, or only one of the two requested plots "
                "(timeseries + correlation) was produced."
            ),
        )
        assert evaluation.passed, (
            f"LLM judge failed the pipeline workflow.\n"
            f"  reasoning: {evaluation.reasoning}\n"
            f"  confidence: {evaluation.confidence}\n"
            f"  warnings: {evaluation.warnings}\n"
            f"  tools used: {result.tool_names}"
        )

        # -- Cost + turn ceilings --
        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

        if result.num_turns is not None:
            assert result.num_turns < 25, (
                f"Test used {result.num_turns} turns — may indicate a loop"
            )

        print(f"  archiver calls: {len(archiver_calls)}")

    # -------------------------------------------------------------------
    # Test 4 — 3D scatter plot via data-visualizer subagent
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_als_apg
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
        project_dir = init_project(tmp_path, "sdk-3d-scatter", provider="als-apg")

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
                f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
            )
            print(f"  tool: {trace.name}{error_flag}{parent_flag}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # archiver_read was called
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # A visualization tool was called (static or interactive)
        viz_calls = result.tools_matching("create_static_plot") + result.tools_matching(
            "create_interactive_plot"
        )
        assert len(viz_calls) > 0, f"No visualization tool called. Tools used: {result.tool_names}"

        # The viz tool should have succeeded (no errors)
        viz_errors = [t for t in viz_calls if t.is_error]
        assert len(viz_errors) == 0, (
            f"Visualization tool returned errors: {[t.result[:300] for t in viz_errors]}"
        )

        # Check that the viz code contains 3D-related keywords
        has_3d_code = False
        for call in viz_calls:
            code = call.input.get("code", "")
            if any(
                kw in code.lower()
                for kw in [
                    "3d",
                    "scatter3d",
                    "mplot3d",
                    "axes3d",
                    "projection",
                ]
            ):
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
