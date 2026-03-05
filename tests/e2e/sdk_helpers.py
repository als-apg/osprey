"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and project initialization
utilities used by both the functional SDK tests and the safety E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from osprey.cli.init_cmd import init

# SDK imports — skip entire module if not installed
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        PermissionResultAllow,
        PermissionResultDeny,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolPermissionContext,
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
    model: str = "haiku",
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


def combined_text(result: SDKWorkflowResult) -> str:
    """Combine all text blocks and tool results into a single searchable string."""
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()


def find_png_files(root: Path) -> list[Path]:
    """Recursively find all .png files under *root*."""
    return sorted(root.rglob("*.png"))


def find_html_files(root: Path) -> list[Path]:
    """Recursively find all .html files under *root*, excluding index.html."""
    return sorted(p for p in root.rglob("*.html") if p.name != "index.html")


def read_audit_events(project_dir: Path) -> list[dict]:
    """Read OSPREY tool-call events from Claude Code native transcripts.

    Uses TranscriptReader to extract events from the most recent transcript
    in ``~/.claude/projects/<encoded>/``.

    Returns:
        List of event dicts (tool_call, agent_start, agent_stop).
    """
    from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

    reader = TranscriptReader(project_dir)
    return reader.read_current_session()


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
                                matched.result = "\n".join(texts) if texts else str(block.content)
                            matched.is_error = bool(block.is_error)

            elif isinstance(message, SystemMessage):
                workflow.system_messages.append(message)

            elif isinstance(message, ResultMessage):
                workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    return workflow


# ---------------------------------------------------------------------------
# Hook-observed SDK runner (uses can_use_tool callback)
# ---------------------------------------------------------------------------


@dataclass
class HookEvent:
    """Record of a permission callback invocation (hook returned 'ask')."""

    tool_name: str
    tool_input: dict
    decision: str  # "allow" or "deny"
    reason: str | None = None


@dataclass
class HookObservedResult(SDKWorkflowResult):
    """Extends SDKWorkflowResult with hook observability."""

    hook_events: list[HookEvent] = field(default_factory=list)


async def run_sdk_query_with_hooks(
    project_dir: Path,
    prompt: str,
    *,
    approval_policy: Callable[[str, dict[str, Any]], bool] | str = "auto_approve",
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str = "anthropic/claude-haiku",
) -> HookObservedResult:
    """Run a query via the Claude Agent SDK with hooks enabled and can_use_tool callback.

    Unlike ``run_sdk_query`` (which uses bypassPermissions), this function uses
    ``permission_mode="default"`` so that file-system hooks actually execute.
    When a hook returns ``permissionDecision: "ask"``, the ``can_use_tool``
    callback is invoked instead of prompting a human.

    The ``approval_policy`` controls what happens when a hook returns "ask":
    - ``"auto_approve"`` — always approve (hooks still run, but "ask" → allow)
    - ``"auto_deny"`` — always deny (test that denial propagates correctly)
    - callable — custom ``(tool_name, tool_input) -> bool`` for fine-grained control

    Every callback invocation is recorded in ``hook_events`` for observability.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        approval_policy: How to handle "ask" decisions from hooks.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use (defaults to Haiku for cost-effectiveness).

    Returns:
        HookObservedResult with tool traces, text, metadata, and hook events.
    """
    hook_events: list[HookEvent] = []
    stderr_lines: list[str] = []

    async def _can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Permission callback: record the event and apply the approval policy."""
        if approval_policy == "auto_approve":
            should_allow = True
        elif approval_policy == "auto_deny":
            should_allow = False
        elif callable(approval_policy):
            should_allow = approval_policy(tool_name, tool_input)
        else:
            raise ValueError(f"Invalid approval_policy: {approval_policy!r}")

        decision = "allow" if should_allow else "deny"
        event = HookEvent(
            tool_name=tool_name,
            tool_input=tool_input,
            decision=decision,
            reason=f"approval_policy={approval_policy!r}"
            if isinstance(approval_policy, str)
            else "custom_policy",
        )
        hook_events.append(event)

        if should_allow:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Denied by test approval policy")

    options = ClaudeAgentOptions(
        model=model,
        cwd=str(project_dir),
        permission_mode="default",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=sdk_env(),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        can_use_tool=_can_use_tool,
    )

    workflow = HookObservedResult()

    # Map tool_use_id → ToolTrace for matching results to calls
    pending_tools: dict[str, ToolTrace] = {}

    try:
        # ClaudeSDKClient is required for can_use_tool (streaming mode).
        # The simple query() function does not support permission callbacks.
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
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
                            matched = pending_tools.get(block.tool_use_id)
                            if matched:
                                if isinstance(block.content, str):
                                    matched.result = block.content
                                elif isinstance(block.content, list):
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
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    workflow.hook_events = hook_events
    return workflow
