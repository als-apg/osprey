"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and project initialization
utilities used by both the functional SDK tests and the safety E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
    channel_finder_mode: str | None = None,
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
    if channel_finder_mode is not None:
        args.extend(["--channel-finder-mode", channel_finder_mode])
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
