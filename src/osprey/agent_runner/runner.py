"""Single-prompt agent runs.

``run_query`` sends one prompt and returns the collected
:class:`~osprey.agent_runner.primitives.SDKWorkflowResult`; it is the entry
point ``osprey query`` uses. ``stream_query`` sends one prompt and yields the
run's plain :mod:`~osprey.agent_runner.events` records as they arrive, for a
caller that renders or forwards output live. The multi-turn counterpart is
``osprey.agent_runner.session.agent_session``. All of them build their SDK
options through ``primitives.build_agent_options`` and wait for MCP servers
through ``primitives._ready_mcp``, so provider routing, the readiness barrier
and message handling are identical across them.

The module remains importable when ``claude_agent_sdk`` is absent; the runtime
path will raise ``ImportError`` in that case, but module-level imports (e.g.
for type checking or CLI argument parsing) still succeed.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_agent_sdk import (
        CanUseTool,
        ClaudeAgentOptions,
        HookCallback,
        McpServerConfig,
        PermissionMode,
        SettingSource,
    )
    from claude_agent_sdk.types import SystemPromptFile, SystemPromptPreset

    from osprey.agent_runner.events import AgentEvent

# SDK import — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeSDKError

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from osprey.agent_runner.errors import AgentRunError
from osprey.agent_runner.events import translate_message
from osprey.agent_runner.primitives import (
    SDKWorkflowResult,
    ToolTrace,
    _absorb_message,
    _ready_mcp,
    _send_turn,
    build_agent_options,
)


async def _query_messages(
    options: ClaudeAgentOptions,
    project_dir: Path,
    prompt: str | Sequence[Mapping[str, Any]],
    *,
    await_mcp_servers: Collection[str] | None,
    require_mcp_servers: Collection[str],
    on_mcp_status: Callable[[list[dict[str, Any]]], None] | None,
) -> AsyncGenerator[object, None]:
    """Open a client, wait for MCP, send *prompt* and yield the raw response messages.

    The streaming ``ClaudeSDKClient`` rather than the one-shot ``query()``,
    because only the client exposes ``get_mcp_status()`` for the readiness
    barrier. Closing this generator early (``aclose()``) unwinds the client.

    Raises:
        AgentRunError: For any agent SDK error, chained from it.
        McpNotReadyError: When a required MCP server is not connected.
    """
    try:
        async with ClaudeSDKClient(options=options) as client:
            await _ready_mcp(
                client,
                project_dir,
                await_mcp_servers=await_mcp_servers,
                require_mcp_servers=require_mcp_servers,
                on_mcp_status=on_mcp_status,
            )
            await _send_turn(client, prompt)
            async for message in client.receive_response():
                yield message
    except ClaudeSDKError as exc:
        raise AgentRunError(str(exc), error_type=type(exc).__name__) from exc


def _require_sdk(entry_point: str) -> None:
    if not HAS_SDK:
        raise ImportError(
            f"claude_agent_sdk is required for {entry_point}. "
            "Install it with: pip install claude-agent-sdk"
        )


async def stream_query(
    project_dir: Path,
    prompt: str | Sequence[Mapping[str, Any]],
    *,
    disallowed_tools: Sequence[str],
    max_turns: int | None = 25,
    max_budget_usd: float | None = 2.0,
    model: str | None = None,
    permission_mode: PermissionMode | None = "bypassPermissions",
    setting_sources: list[SettingSource] | None = None,
    allowed_tools: Sequence[str] = (),
    system_prompt: str | SystemPromptPreset | SystemPromptFile | None = None,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    mcp_servers: Mapping[str, McpServerConfig] | Path | None = None,
    session_id: str | None = None,
    resume: str | None = None,
    can_use_tool: CanUseTool | None = None,
    pre_tool_use_hooks: Sequence[HookCallback] = (),
    stderr: Callable[[str], None] | None = None,
    await_mcp_servers: Collection[str] | None = None,
    require_mcp_servers: Collection[str] = (),
    on_mcp_status: Callable[[list[dict[str, Any]]], None] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one prompt and yield the run's event records as they arrive.

    The agent options are built by ``build_agent_options`` from the keywords of
    the same names; see it for each one. The stream ends after the
    :class:`~osprey.agent_runner.events.ResultEvent`.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user turn: text, or a sequence of content blocks (text,
            images, …) sent as one user message.
        await_mcp_servers: The MCP servers to wait for before the prompt is
            sent; ``None`` for the project's declared ones, empty to skip the
            wait.
        require_mcp_servers: Servers without which the run is refused rather
            than started.
        on_mcp_status: Receives the MCP status snapshot the wait ended on,
            before any refusal.

    Yields:
        The run's records, in stream order.

    Raises:
        ImportError: When ``claude_agent_sdk`` is not installed.
        AgentRunError: When the agent SDK fails mid-run.
        McpNotReadyError: When a required MCP server is not connected.
    """
    _require_sdk("stream_query")
    options = build_agent_options(
        project_dir,
        disallowed_tools=disallowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        permission_mode=permission_mode,
        setting_sources=setting_sources,
        allowed_tools=allowed_tools,
        system_prompt=system_prompt,
        env=env,
        provider=provider,
        mcp_servers=mcp_servers,
        session_id=session_id,
        resume=resume,
        can_use_tool=can_use_tool,
        pre_tool_use_hooks=pre_tool_use_hooks,
        stderr=stderr,
    )
    # aclosing: closing this stream (or cancelling it at a yield) must unwind
    # the client and its CLI child before aclose() returns, not at finalization.
    async with contextlib.aclosing(
        _query_messages(
            options,
            project_dir,
            prompt,
            await_mcp_servers=await_mcp_servers,
            require_mcp_servers=require_mcp_servers,
            on_mcp_status=on_mcp_status,
        )
    ) as messages:
        async for message in messages:
            for event in translate_message(message):
                yield event


async def run_query(
    project_dir: Path,
    prompt: str,
    *,
    disallowed_tools: Sequence[str],
    max_turns: int | None = 25,
    max_budget_usd: float | None = 2.0,
    model: str | None = None,
    permission_mode: PermissionMode | None = "bypassPermissions",
    setting_sources: list[SettingSource] | None = None,
    allowed_tools: Sequence[str] = (),
    system_prompt: str | SystemPromptPreset | SystemPromptFile | None = None,
    env: Mapping[str, str] | None = None,
    provider: str | None = None,
    mcp_servers: Mapping[str, McpServerConfig] | Path | None = None,
    session_id: str | None = None,
    resume: str | None = None,
    can_use_tool: CanUseTool | None = None,
    pre_tool_use_hooks: Sequence[HookCallback] = (),
    stderr: Callable[[str], None] | None = None,
    await_mcp_servers: Collection[str] | None = None,
) -> SDKWorkflowResult:
    """Run a single-turn agent query via the Claude Agent SDK.

    This is the production runner used by ``osprey query``.  It is
    architecturally read-only: the caller supplies ``disallowed_tools`` which
    the SDK forwards to the Claude Code CLI as ``--disallowedTools``, blocking
    writes even under ``permission_mode=bypassPermissions``.

    The first turn waits for the MCP servers to register (see
    ``primitives.await_mcp_ready``) so the agent starts with a fully registered
    toolset. The agent options are built by ``build_agent_options`` from the
    keywords of the same names; see it for each one.

    Args:
        project_dir: Path to an initialized OSPREY project.
        prompt: The user prompt to send to the agent.
        disallowed_tools: Tool names forbidden at the SDK level.  This is the
            architectural read-only guard; the caller is responsible for
            supplying the appropriate list (see
            ``.claude/hooks/hook_config.json`` ``write_tools``).
        max_turns: Maximum agentic turns before stopping.
        max_budget_usd: Budget cap in USD (not scaled — this is the literal
            ceiling passed to the SDK).
        model: Model identifier.  When ``None``, the project's main model via
            ``resolve_default_model``.
        await_mcp_servers: The MCP servers to wait for before the prompt is
            sent; ``None`` for the project's declared ones, empty to skip the
            wait.

    Returns:
        SDKWorkflowResult with all collected tool traces, text blocks,
        system messages, MCP server snapshot, and the final ResultMessage.

    Raises:
        ImportError: When ``claude_agent_sdk`` is not installed.
        RuntimeError: When the underlying SDK query fails.
    """
    _require_sdk("run_query")

    options = build_agent_options(
        project_dir,
        disallowed_tools=disallowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        permission_mode=permission_mode,
        setting_sources=setting_sources,
        allowed_tools=allowed_tools,
        system_prompt=system_prompt,
        env=env,
        provider=provider,
        mcp_servers=mcp_servers,
        session_id=session_id,
        resume=resume,
        can_use_tool=can_use_tool,
        pre_tool_use_hooks=pre_tool_use_hooks,
        stderr=stderr,
    )

    workflow = SDKWorkflowResult()
    pending_tools: dict[str, ToolTrace] = {}

    def _keep_snapshot(servers: list[dict[str, Any]]) -> None:
        workflow.mcp_servers = servers

    try:
        async for message in _query_messages(
            options,
            project_dir,
            prompt,
            await_mcp_servers=await_mcp_servers,
            require_mcp_servers=(),
            on_mcp_status=_keep_snapshot,
        ):
            _absorb_message(message, workflow, pending_tools)
    except Exception as exc:
        raise RuntimeError(f"SDK query failed: {exc}") from exc

    return workflow
