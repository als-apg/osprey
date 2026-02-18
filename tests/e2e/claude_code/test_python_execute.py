"""E2E tests for python_execute functional workflows via Claude Code SDK.

Tests the full pipeline: user prompt -> Claude -> MCP tool call -> hooks ->
adapter -> execution -> response.

These tests verify that python_execute works correctly for non-safety
scenarios (basic execution, matplotlib, archiver pipelines).

Requires:
- Claude Code CLI installed
- claude_agent_sdk Python package installed
- ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import (
    init_project,
    run_sdk_query,
)


class TestPythonExecuteE2E:
    """E2E functional tests for python_execute through the Claude Code SDK."""

    # -------------------------------------------------------------------
    # Test 1 — Readonly smoke test
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_python_execute_readonly_smoke(self, tmp_path):
        """Simple print() code runs and returns stdout via python_execute tool.

        Verifies the full tool pipeline works end-to-end: Claude receives
        the prompt, calls python_execute with readonly code, and gets stdout.

        Cost budget: $0.25
        """
        project_dir = init_project(tmp_path, "pyexec-smoke")

        prompt = (
            "Use the python_execute tool to run this exact code: "
            "print(6 * 7). "
            "Report the output you get."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=5,
            max_budget_usd=0.25,
        )

        # -- Debug output --
        print("\n--- python_execute readonly smoke ---")
        print(f"  tools called: {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            print(f"  tool: {trace.name}")
            print(f"    is_error: {trace.is_error}")
            result_preview = (trace.result or "")[:300]
            print(f"    result preview: {result_preview}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # python_execute should have been called
        py_tools = result.tools_matching("python_execute")
        assert len(py_tools) >= 1, (
            f"Expected python_execute call but got: {result.tool_names}"
        )

        # The tool call should not have errored
        assert not py_tools[0].is_error, (
            f"python_execute returned error: {py_tools[0].result}"
        )

        # "42" should appear in the tool result or assistant text
        all_text = " ".join(result.text_blocks)
        tool_results = " ".join(t.result or "" for t in py_tools)
        assert "42" in all_text or "42" in tool_results, (
            f"Expected '42' in output.\n  Text: {all_text[:500]}\n  Tool: {tool_results[:500]}"
        )

    # -------------------------------------------------------------------
    # Test 2 — Matplotlib figure creation
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_python_execute_with_matplotlib(self, tmp_path):
        """Code that creates a matplotlib plot runs successfully.

        Verifies that python_execute can handle matplotlib code and that
        the execution completes without error.

        Cost budget: $0.50
        """
        project_dir = init_project(tmp_path, "pyexec-matplotlib")

        prompt = (
            "Use the python_execute tool to run this code:\n\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "x = np.linspace(0, 10, 100)\n"
            "plt.figure()\n"
            "plt.plot(x, np.sin(x))\n"
            "plt.title('Sine Wave')\n"
            "plt.savefig('sine_plot.png')\n"
            "print('Plot saved successfully')\n\n"
            "Report the output."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=10,
            max_budget_usd=0.50,
        )

        # -- Debug output --
        print("\n--- python_execute matplotlib ---")
        print(f"  tools called: {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"

        # python_execute should have been called
        py_tools = result.tools_matching("python_execute")
        assert len(py_tools) >= 1, (
            f"Expected python_execute call but got: {result.tool_names}"
        )

        # Check that execution succeeded (tool result mentions success or plot saved)
        tool_results = " ".join(t.result or "" for t in py_tools)
        all_text = " ".join(result.text_blocks)
        combined = (tool_results + " " + all_text).lower()
        assert "success" in combined or "plot saved" in combined or "saved" in combined, (
            f"Expected success indicator.\n  Tool: {tool_results[:500]}\n  Text: {all_text[:500]}"
        )

    # -------------------------------------------------------------------
    # Test 3 — Archiver read + python_execute pipeline
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_archiver_then_python_pipeline(self, tmp_path):
        """archiver_read -> python_execute pipeline works with the adapter.

        Verifies that the adapter doesn't break the existing workflow where
        Claude reads archiver data and then uses python_execute to process it.

        Cost budget: $1.00
        """
        project_dir = init_project(tmp_path, "pyexec-pipeline")

        prompt = (
            "I want to analyze recent beam current data. Please:\n"
            "1. Use archiver_read to get the last 1 hour of data for "
            "SR:C01-BI:G02A{BPM:1}SA:X-I\n"
            "2. Then use python_execute to compute the mean and standard "
            "deviation of the values.\n"
            "Report the statistics."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=15,
            max_budget_usd=1.00,
        )

        # -- Debug output --
        print("\n--- archiver+python pipeline ---")
        print(f"  tools called: {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            print(f"  tool: {trace.name}")
            print(f"    is_error: {trace.is_error}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"

        # Both tools should have been called
        archiver_tools = result.tools_matching("archiver_read")
        py_tools = result.tools_matching("python_execute")

        assert len(archiver_tools) >= 1, (
            f"Expected archiver_read call but got: {result.tool_names}"
        )
        assert len(py_tools) >= 1, (
            f"Expected python_execute call but got: {result.tool_names}"
        )

        # The python_execute call should not have errored
        for pt in py_tools:
            assert not pt.is_error, f"python_execute returned error: {pt.result}"
