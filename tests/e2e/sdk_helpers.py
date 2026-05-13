"""Shared helpers for Claude Code SDK-based E2E tests.

Provides the SDK runner, tool-trace dataclasses, and project initialization
utilities used by both the functional SDK tests and the safety E2E tests.

Extracted from test_claude_code_sdk_e2e.py to avoid circular imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        UserMessage,
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


def has_als_apg_api_key() -> bool:
    """Check if ALS_APG_API_KEY is set.

    The CI-default Bedrock proxy at llm.gianlucamartino.com authenticates
    with this token; the safety/SDK E2E suite skip-gates on it because the
    Claude Code CLI subprocess uses the proxy via the project's `.env` and
    `provider=als-apg` defaults landed in 8c541cc9.
    """
    return bool(os.environ.get("ALS_APG_API_KEY"))


def init_project(
    tmp_path: Path,
    name: str,
    template: str = "control_assistant",
    *,
    provider: str,
    model: str = "haiku",
    channel_finder_mode: str | None = None,
    tier: int = 1,
) -> Path:
    """Create a project via ``osprey build --preset <template>``, return project_dir.

    ``provider`` is required (keyword-only) — every test callsite must name
    it explicitly. Each provider gates on different credentials (CBORG needs
    LBLnet/VPN; als-apg needs ``ALS_APG_API_KEY``; anthropic-direct needs
    ``ANTHROPIC_API_KEY``), so a kwarg default silently couples tests to one
    provider's auth and produces the local-passes-CI-fails asymmetry. Pick
    ``"als-apg"`` for GitHub Actions runners, ``"cborg"`` from LBLnet, or
    ``"anthropic"`` when you have an ``ANTHROPIC_API_KEY`` available.

    Invoked via ``subprocess`` rather than Click's ``CliRunner`` because
    ``osprey build`` instantiates ``rich.Console(force_terminal=True)``,
    which performs terminal-aware lifecycle management on the captured
    ``BytesIO`` stream that ``CliRunner`` substitutes for stdout. On
    Python ≥3.11 that closes the wrapper before Click reads it back,
    raising ``ValueError: I/O operation on closed file`` at fixture
    setup. ``CliRunner`` is also a unit-test harness; an e2e fixture
    should exercise the same entry point real users invoke.
    """
    args = [
        name,
        "--preset",
        template.replace("_", "-"),
        "--skip-deps",
        "--skip-lifecycle",
        "--output-dir",
        str(tmp_path),
        "--tier",
        str(tier),
        "--set",
        f"provider={provider}",
        "--set",
        f"model={model}",
    ]
    if channel_finder_mode is not None:
        args.extend(["--set", f"channel_finder_mode={channel_finder_mode}"])
    result = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "build", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"osprey build failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    project_dir = tmp_path / name
    assert project_dir.exists(), f"Project directory not created: {project_dir}"
    return project_dir


def enable_writes_in_project(project_dir: Path) -> None:
    """Ensure ``control_system.writes_enabled`` is true in the project's config.yml.

    Required for tests that exercise the approval-hook path: the
    ``osprey_writes_check.py`` PreToolUse hook denies before
    ``osprey_approval.py`` gets to return ``ask``, so the SDK's
    ``can_use_tool`` callback never fires when writes are disabled.

    Idempotent: presets like ``control_assistant`` already ship with
    ``writes_enabled: true``; only ``hello_world`` defaults to false.
    """
    config_path = project_dir / "config.yml"
    text = config_path.read_text(encoding="utf-8")
    if "writes_enabled: true" in text:
        return
    updated = text.replace("writes_enabled: false", "writes_enabled: true", 1)
    if updated == text:
        raise RuntimeError(
            f"Could not enable writes in {config_path}: no writes_enabled key found."
        )
    config_path.write_text(updated, encoding="utf-8")


def _resolve_project_spec(project_dir: Path):
    """Return the project's ``ClaudeCodeModelSpec`` or ``None`` when the
    project's ``config.yml`` has no ``claude_code`` block.

    Reads ``config.yml`` and runs the same resolver ``osprey claude chat``
    uses, so test routing matches production exactly. Unlike the earlier
    bare-``except`` version, this surfaces any unexpected error (missing
    ``config.yml``, YAML parse failure, resolver import failure) rather
    than masking it as ``None``.
    """
    import yaml

    from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

    cfg = yaml.safe_load((project_dir / "config.yml").read_text()) or {}
    return ClaudeCodeModelResolver.resolve(
        cfg.get("claude_code", {}),
        cfg.get("api", {}).get("providers", {}),
    )


def provider_env_for_project(project_dir: Path) -> dict[str, str]:
    """Resolve provider env vars so the SDK routes to the project's provider.

    Without this, the bundled Claude CLI defaults to ``api.anthropic.com``
    using whatever ambient ``ANTHROPIC_API_KEY`` happens to be set — which
    is the wrong endpoint for cborg/als-apg projects and 404s on model
    aliases like ``anthropic/claude-haiku``.

    Returns the spec's ``env_block`` (``ANTHROPIC_BASE_URL``,
    ``ANTHROPIC_DEFAULT_*_MODEL``, ...) plus the auth var populated from
    the configured shell secret. Raises ``RuntimeError`` if the project
    has no resolvable provider — silently falling back to ``{}`` would
    let the test subprocess inherit ambient env, which is the exact
    local-vs-CI divergence we are trying to eliminate.
    """
    spec = _resolve_project_spec(project_dir)
    if spec is None:
        raise RuntimeError(
            f"Project at {project_dir} has no resolvable provider in "
            "config.yml — pass provider=<als-apg|cborg|anthropic|amsc|argo> "
            "to init_project()."
        )
    env: dict[str, str] = dict(spec.env_block)
    if spec.auth_secret_env and spec.auth_env_var:
        secret = os.environ.get(spec.auth_secret_env)
        if secret:
            env[spec.auth_env_var] = secret
    return env


def _default_haiku_model(project_dir: Path) -> str:
    """Resolve the project's haiku-tier model name."""
    spec = _resolve_project_spec(project_dir)
    if spec is not None:
        return spec.tier_to_model.get("haiku", "claude-haiku-4-5-20251001")
    return "claude-haiku-4-5-20251001"


def sdk_env(project_dir: Path | None = None) -> dict[str, str]:
    """Return env overrides for the SDK subprocess.

    Always sets ``CLAUDECODE=""`` to bypass the nested-session guard
    (JavaScript treats "" as falsy). When ``project_dir`` is provided,
    also injects the project's resolved provider env block so the bundled
    CLI talks to the configured provider (cborg, als-apg, anthropic-direct)
    instead of falling through to ``api.anthropic.com``.
    """
    env = {"CLAUDECODE": ""}
    if project_dir is not None:
        env.update(provider_env_for_project(project_dir))
    return env


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


def _ingest_tool_result(
    block: ToolResultBlock, pending_tools: dict[str, ToolTrace]
) -> None:
    """Match a ToolResultBlock to its pending ToolTrace and populate result/is_error.

    ToolResultBlocks arrive in ``UserMessage.content`` (per Anthropic API contract:
    tool_use is assistant output, tool_result is user input back to the model).
    They may also appear in ``AssistantMessage`` when the SDK forwards them.
    """
    matched = pending_tools.get(block.tool_use_id)
    if matched is None:
        return
    if isinstance(block.content, str):
        matched.result = block.content
    elif isinstance(block.content, list):
        texts = []
        for item in block.content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        matched.result = "\n".join(texts) if texts else str(block.content)
    matched.is_error = bool(block.is_error)


async def run_sdk_query(
    project_dir: Path,
    prompt: str,
    *,
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str | None = None,
    disallowed_tools: list[str] | None = None,
) -> SDKWorkflowResult:
    """Run a query via the Claude Agent SDK and collect full tool traces.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send.
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD.
        model: Model to use. Defaults to the project's haiku-tier model
            resolved from ``config.yml`` (e.g. ``claude-haiku-4-5`` for
            cborg, ``claude-haiku-4-5-20251001`` for direct anthropic).
        disallowed_tools: Optional list of tool names to forbid at the SDK
            level. Forwarded to the Claude Code CLI as ``--disallowedTools``,
            which takes precedence over ``permission_mode=bypassPermissions``
            and over per-tool ``permissions_allow`` in ``.mcp.json``. Use this
            to architecturally force delegation to subagents (the main agent
            cannot call a disallowed tool even when settings would permit it).

    Returns:
        SDKWorkflowResult with all collected tool traces, text, and metadata.
    """
    # Collect stderr lines for debugging CLI failures
    stderr_lines: list[str] = []

    options = ClaudeAgentOptions(
        model=model if model is not None else _default_haiku_model(project_dir),
        cwd=str(project_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=sdk_env(project_dir),
        stderr=lambda line: stderr_lines.append(line),
        setting_sources=["project"],
        disallowed_tools=disallowed_tools or [],
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
                        _ingest_tool_result(block, pending_tools)

            elif isinstance(message, UserMessage):
                # Tool results land here per the Anthropic API contract.
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            _ingest_tool_result(block, pending_tools)

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
    model: str | None = None,
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
        model: Model to use. Defaults to the project's haiku-tier model
            resolved from ``config.yml``.

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
        model=model if model is not None else _default_haiku_model(project_dir),
        cwd=str(project_dir),
        permission_mode="default",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=sdk_env(project_dir),
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
                            _ingest_tool_result(block, pending_tools)

                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                _ingest_tool_result(block, pending_tools)

                elif isinstance(message, SystemMessage):
                    workflow.system_messages.append(message)

                elif isinstance(message, ResultMessage):
                    workflow.result = message
    except Exception as exc:
        stderr_output = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
        raise RuntimeError(f"SDK query failed: {exc}\n\nCLI stderr:\n{stderr_output}") from exc

    workflow.hook_events = hook_events
    return workflow
