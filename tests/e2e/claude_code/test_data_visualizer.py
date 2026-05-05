"""E2E tests for the data-visualizer sub-agent.

Tests the full pipeline: archiver_read → data-visualizer agent →
create_interactive_plot / create_static_plot / create_dashboard → artifact.

These tests use real API calls via the Claude Agent SDK.

Requires:
- Claude Code CLI installed
- claude_agent_sdk Python package installed
- ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import (
    find_html_files,
    init_project,
    run_sdk_query,
)


class TestDataVisualizerAgent:
    """E2E tests for the data-visualizer sub-agent."""

    # -------------------------------------------------------------------
    # Test 1 — Interactive plot with data_source parameter
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_interactive_plot_with_data_source(self, tmp_path):
        """Regression: data-visualizer should use data_source without errors.

        Reproduces a real failure where the agent:
        1. Passed a context entry ID ("2") as data_source → FileNotFoundError
           (only artifact IDs and file paths are supported)
        2. Passed a file path but wrote code assuming raw dict structure →
           KeyError because build_data_reader() auto-unwraps the JSON
           envelope before the agent's code runs

        The agent should be able to pass data_source (file path or context
        entry ID) and have create_interactive_plot load the data correctly
        on the FIRST attempt.

        Pipeline: archiver_read → data-visualizer → create_interactive_plot
        """
        project_dir = init_project(tmp_path, "e2e-viz-data-source")

        prompt = (
            "Use archiver_read to retrieve data for channels "
            "'DIAG:BPM[BPM18]:POSITION:X', 'DIAG:BPM[BPM19]:POSITION:X', "
            "'DIAG:BPM[BPM20]:POSITION:X' over the last 24 hours with "
            "processing 'mean' and bin_size 60. "
            "Then delegate to the data-visualizer agent to create an "
            "interactive Plotly scatter plot of BPM18 vs BPM19, colored by "
            "time. The data-visualizer should use the data_source parameter "
            "of create_interactive_plot to load the archiver data — do NOT "
            "hardcode file paths in the plotting code."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=25,
            max_budget_usd=2.0,
        )

        # -- Debug output --
        print("\n--- data-visualizer data_source regression test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            error_flag = " [ERROR]" if trace.is_error else ""
            parent_flag = (
                f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
            )
            print(f"  tool: {trace.name}{error_flag}{parent_flag}")
            if trace.is_error and trace.result:
                print(f"    error: {trace.result[:300]}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # archiver_read was called
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # create_interactive_plot was called (not static)
        viz_calls = result.tools_matching("create_interactive_plot")
        assert len(viz_calls) > 0, (
            f"create_interactive_plot not called. Tools used: {result.tool_names}"
        )

        # REGRESSION CRITERION: The first create_interactive_plot call should
        # succeed. If data_source handling is broken, the agent will fail on
        # the first attempt and retry with workarounds.
        first_viz_call = viz_calls[0]
        assert not first_viz_call.is_error, (
            f"First create_interactive_plot call failed (data_source bug). "
            f"Error: {first_viz_call.result[:500] if first_viz_call.result else 'N/A'}"
        )

        # The successful call should have used data_source parameter
        viz_input = first_viz_call.input
        assert viz_input.get("data_source"), (
            "create_interactive_plot was called without data_source — "
            "the agent should be passing archiver data via data_source, "
            f"not hardcoding file reads. Input keys: {list(viz_input.keys())}"
        )

        # No viz errors at all (no retries needed)
        viz_errors = [t for t in viz_calls if t.is_error]
        assert len(viz_errors) == 0, (
            f"{len(viz_errors)} create_interactive_plot error(s) — "
            f"agent needed retries: {[t.result[:200] for t in viz_errors if t.result]}"
        )

        # HTML artifact exists in the workspace
        html_files = find_html_files(project_dir / "_agent_data")
        assert len(html_files) > 0, (
            "No HTML artifact found in _agent_data — "
            "create_interactive_plot may not have produced a Plotly chart."
        )

        # Cost should be under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

        print(f"  viz calls: {len(viz_calls)} (errors: {len(viz_errors)})")
        print(f"  HTML files: {[p.name for p in html_files]}")

    # -------------------------------------------------------------------
    # Test 2 — Interactive plot with context entry ID as data_source
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.asyncio
    async def test_interactive_plot_context_entry_id(self, tmp_path):
        """Regression: data_source should accept context entry IDs.

        The archiver_read tool returns a context_entry_id (e.g., 2) in its
        response. The agent naturally passes this as data_source, but
        build_data_reader() only recognizes 12-char hex artifact IDs and
        file paths — integer IDs cause FileNotFoundError.

        This test verifies the agent can reference archiver data by its
        context entry ID.

        Pipeline: archiver_read → data-visualizer → create_interactive_plot
        """
        project_dir = init_project(tmp_path, "e2e-viz-context-id")

        prompt = (
            "Use archiver_read to retrieve data for channels "
            "'DIAG:BPM[BPM18]:POSITION:X' and 'DIAG:BPM[BPM19]:POSITION:X' "
            "over the last 4 hours with processing 'mean' and bin_size 300. "
            "Then delegate to the data-visualizer agent to create an "
            "interactive Plotly line chart of both BPMs over time. "
            "Tell the data-visualizer to reference the archiver data by its "
            "context entry ID when calling create_interactive_plot."
        )

        result = await run_sdk_query(
            project_dir,
            prompt,
            max_turns=25,
            max_budget_usd=2.0,
        )

        # -- Debug output --
        print("\n--- data-visualizer context entry ID test ---")
        print(f"  tools called ({len(result.tool_traces)}): {result.tool_names}")
        print(f"  num_turns: {result.num_turns}")
        print(f"  cost: ${result.cost_usd:.4f}" if result.cost_usd else "  cost: N/A")
        for trace in result.tool_traces:
            error_flag = " [ERROR]" if trace.is_error else ""
            print(f"  tool: {trace.name}{error_flag}")
            if trace.is_error and trace.result:
                print(f"    error: {trace.result[:300]}")

        # -- Assertions --
        assert result.result is not None, "No ResultMessage received"
        assert not result.result.is_error, f"SDK query ended in error: {result.result.result}"

        # archiver_read was called
        archiver_calls = result.tools_matching("archiver_read")
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # A visualization tool was called
        viz_calls = result.tools_matching("create_interactive_plot")
        assert len(viz_calls) > 0, (
            f"create_interactive_plot not called. Tools used: {result.tool_names}"
        )

        # No viz errors — if context entry ID resolution works, first try succeeds
        viz_errors = [t for t in viz_calls if t.is_error]
        assert len(viz_errors) == 0, (
            f"{len(viz_errors)} create_interactive_plot error(s): "
            f"{[t.result[:200] for t in viz_errors if t.result]}"
        )

        # REGRESSION CRITERION: at least one viz call must have used a
        # context-entry ID (small integer, or its string repr) as data_source.
        # The bug under test was that build_data_reader() rejected integer
        # IDs — accepting a file path or 12-char hex artifact ID would
        # leave the regression unverified.
        def _looks_like_context_entry_id(ds: object) -> bool:
            if isinstance(ds, int):
                return True
            if isinstance(ds, str):
                stripped = ds.strip()
                # Plain int repr like "2" or "17" — exclude artifact IDs
                # (12-char hex) and file paths (contain "/" or ".").
                return stripped.isdigit() and len(stripped) <= 6
            return False

        data_sources = [call.input.get("data_source") for call in viz_calls]
        print(f"  data_source values: {data_sources!r}")
        assert any(_looks_like_context_entry_id(ds) for ds in data_sources), (
            "No create_interactive_plot call used a context-entry ID as data_source. "
            "The agent may have fallen back to file paths or artifact IDs, which "
            "would not exercise the regression target. "
            f"data_source values seen: {data_sources!r}"
        )

        # HTML artifact produced
        html_files = find_html_files(project_dir / "_agent_data")
        assert len(html_files) > 0, "No HTML artifact found — interactive plot not produced."

        if result.cost_usd is not None:
            assert result.cost_usd < 2.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $2.00 budget"
            )

        print(f"  viz calls: {len(viz_calls)} (errors: {len(viz_errors)})")
        print(f"  HTML files: {[p.name for p in html_files]}")
