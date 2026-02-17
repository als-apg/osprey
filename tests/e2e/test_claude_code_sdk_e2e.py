"""SDK-based E2E tests for Claude Code + OSPREY MCP integration.

These tests use the Claude Agent SDK (claude_agent_sdk) to spawn Claude Code
as a subprocess and stream typed Python objects for every step, enabling
assertions on individual tool calls, their inputs, outputs, ordering, and cost.

This is an evolution of the subprocess-based tests in
test_claude_code_init_integration.py — same workflows, but with full
tool-level observability instead of just stdout text.

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

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init

# SDK imports — skip entire module if not installed
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        query,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is installed and functional."""
    import subprocess

    try:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def has_anthropic_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
) -> Path:
    """Create a project via ``osprey init`` CLI, return project_dir."""
    runner = CliRunner()
    args = [
        name,
        "--template",
        template,
        "--output-dir",
        str(tmp_path),
        "--provider",
        provider,
        "--model",
        model,
    ]
    result = runner.invoke(init, args)
    assert result.exit_code == 0, f"osprey init failed: {result.output}"
    project_dir = tmp_path / name
    assert project_dir.exists(), f"Project directory not created: {project_dir}"
    return project_dir


def sdk_env() -> dict[str, str]:
    """Return env overrides to bypass nested-session guard.

    When tests run inside a Claude Code session, the CLAUDECODE env var
    triggers a nested-session guard in the CLI. Setting it to empty string
    bypasses this (JavaScript treats "" as falsy).
    """
    return {"CLAUDECODE": ""}



def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


# ---------------------------------------------------------------------------
# Tool trace dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToolTrace:
    """Lightweight record of a single tool call for observability."""

    name: str
    input: dict
    result: str | None = None
    is_error: bool = False
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class SDKWorkflowResult:
    """Aggregated result from an SDK query run."""

    tool_traces: list[ToolTrace] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    system_messages: list[SystemMessage] = field(default_factory=list)
    result: ResultMessage | None = None

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names that were called."""
        return [t.name for t in self.tool_traces]

    @property
    def cost_usd(self) -> float | None:
        """Total cost from the ResultMessage."""
        return self.result.total_cost_usd if self.result else None

    @property
    def num_turns(self) -> int | None:
        """Number of agentic turns from the ResultMessage."""
        return self.result.num_turns if self.result else None

    def tools_matching(self, substring: str) -> list[ToolTrace]:
        """Return all tool traces whose name contains *substring*."""
        return [t for t in self.tool_traces if substring in t.name]


# ---------------------------------------------------------------------------
# Core SDK runner
# ---------------------------------------------------------------------------


async def run_sdk_query(
    project_dir: Path,
    prompt: str,
    *,
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str = "anthropic/claude-haiku",
) -> SDKWorkflowResult:
    """Run a query via the Claude Agent SDK and collect full tool traces.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use (defaults to Haiku for cost-effectiveness).

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.
    """
    # Collect stderr lines for debugging CLI failures
    stderr_lines: list[str] = []

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(project_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=sdk_env(),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
    )

    workflow = SDKWorkflowResult()

    # Map tool_use_id → ToolTrace for matching results to calls
    pending_tools: dict[str, ToolTrace] = {}

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        workflow.text_blocks.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        trace = ToolTrace(
                            name=block.name,
                            input=block.input,
                            tool_use_id=block.id,
                            parent_tool_use_id=message.parent_tool_use_id,
                        )
                        workflow.tool_traces.append(trace)
                        pending_tools[block.id] = trace
                    elif isinstance(block, ToolResultBlock):
                        # Match result to its tool call
                        matched = pending_tools.get(block.tool_use_id)
                        if matched:
                            if isinstance(block.content, str):
                                matched.result = block.content
                            elif isinstance(block.content, list):
                                # Extract text from content list
                                texts = []
                                for item in block.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        texts.append(item.get("text", ""))
                                matched.result = (
                                    "\n".join(texts) if texts else str(block.content)
                                )
                            matched.is_error = bool(block.is_error)

            elif isinstance(message, SystemMessage):
                workflow.system_messages.append(message)

            elif isinstance(message, ResultMessage):
                workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(
            f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}"
        ) from exc

    return workflow


# ---------------------------------------------------------------------------
# Module-level markers & skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(
        not is_claude_code_available(),
        reason="Claude Code CLI not installed",
    ),
]


# ===========================================================================
# Tests
# ===========================================================================


class TestClaudeCodeSDKIntegration:
    """SDK-based E2E tests with full tool-call observability."""

    # -------------------------------------------------------------------
    # Test 1 — Smoke test: single tool call observability
    # -------------------------------------------------------------------

    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.skipif(not has_anthropic_api_key(), reason="ANTHROPIC_API_KEY not set")
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
    # Test 2 — Archiver + plot (hardcoded channels, no channel-resolver)
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.skipif(not has_anthropic_api_key(), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_archiver_and_plot_via_sdk(self, tmp_path):
        """Verify archiver_read → python_execute pipeline with tool ordering.

        Uses hardcoded channel names to bypass the channel-resolver and
        reduce LLM non-determinism and cost.
        """
        project_dir = init_project(tmp_path, "sdk-archiver-plot")

        prompt = (
            "Use the archiver_read tool to retrieve data for channels "
            "'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', "
            "'DIAG:BPM03:POSITION:X' over the last 24 hours. "
            "Then use python_execute to create a timeseries plot of "
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
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # python_execute was called
        python_calls = result.tools_matching("python_execute")
        assert len(python_calls) > 0, f"python_execute not called. Tools used: {result.tool_names}"

        # archiver_read was called BEFORE python_execute
        archiver_idx = next(
            i for i, t in enumerate(result.tool_traces) if "archiver_read" in t.name
        )
        python_idx = next(i for i, t in enumerate(result.tool_traces) if "python_execute" in t.name)
        assert archiver_idx < python_idx, (
            f"archiver_read (idx={archiver_idx}) should come before "
            f"python_execute (idx={python_idx})"
        )

        # PNG artifact exists in the project tree
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, (
            "No PNG files found in the project — python_execute may not have created a plot."
        )

        # Cost should be under budget
        if result.cost_usd is not None:
            assert result.cost_usd < 1.0, (
                f"Test cost ${result.cost_usd:.4f} — exceeded $1.00 budget"
            )

        print(f"  PNG files: {[p.name for p in png_files]}")

    # -------------------------------------------------------------------
    # Test 3 — Full BPM correlation pipeline (channel-resolver → archiver → plot)
    # -------------------------------------------------------------------

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.requires_anthropic
    @pytest.mark.skipif(not has_anthropic_api_key(), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_full_bpm_correlation_pipeline_via_sdk(self, tmp_path):
        """Full multi-tool pipeline with sub-agent delegation.

        Natural language prompt matching the user's interactive workflow:
        channel-resolver sub-agent → archiver_read → python_execute → artifact.

        This is the SDK equivalent of
        test_claude_code_init_integration.py::test_claude_full_bpm_analysis_pipeline.
        """
        project_dir = init_project(tmp_path, "sdk-bpm-pipeline")

        prompt = (
            "Give me a timeseries and a correlation plot of all horizontal "
            "BPM positions over the last 24 hours. Use the channel_find tool "
            "to discover BPM channels, then archiver_read to get historical "
            "data, then python_execute to create the plots. Save the plots "
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
                f" (sub-agent: {trace.parent_tool_use_id})" if trace.parent_tool_use_id else ""
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
        assert len(archiver_calls) > 0, f"archiver_read not called. Tools used: {result.tool_names}"

        # python_execute was called (should contain plotting code)
        python_calls = result.tools_matching("python_execute")
        assert len(python_calls) > 0, f"python_execute not called. Tools used: {result.tool_names}"

        # Check that python_execute input contains plotting-related code
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
            "python_execute was called but code doesn't contain plot-related keywords"
        )

        # At least one PNG artifact was created
        png_files = find_png_files(project_dir)
        assert len(png_files) > 0, "No PNG files found — python_execute may not have created plots."

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
