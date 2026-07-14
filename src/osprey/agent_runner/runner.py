"""Headless agent-run entry point for the ``osprey query`` CLI command.

Provides a production-ready ``run_query`` coroutine built on the shared
primitives in ``osprey.agent_runner.primitives``.  ``run_query`` is the
single-turn entry point; the multi-turn counterpart is
``osprey.agent_runner.session.agent_session``.  Both build their SDK options
via ``primitives.build_agent_options`` and parse the response stream via
``primitives._drain_response``, so provider routing and message handling are
identical across them.

The module remains importable when ``claude_agent_sdk`` is absent; the runtime
path will raise ``ImportError`` in that case, but module-level imports (e.g.
for type checking or CLI argument parsing) still succeed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient

# SDK import — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import ClaudeSDKClient

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from osprey.agent_runner.primitives import (
    SDKWorkflowResult,
    _await_mcp_ready,
    _drain_response,
    _expected_mcp_servers,
    build_agent_options,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_query(
    project_dir: Path,
    prompt: str,
    *,
    disallowed_tools: list[str],
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
    model: str | None = None,
) -> SDKWorkflowResult:
    """Run a single-turn, read-only agent query via the Claude Agent SDK.

    This is the production runner used by ``osprey query``.  It is
    architecturally read-only: the caller supplies ``disallowed_tools`` which
    the SDK forwards to the Claude Code CLI as ``--disallowedTools``, blocking
    writes even under ``permission_mode=bypassPermissions``.

    The function polls ``_await_mcp_ready`` before sending the prompt so the
    agent always starts with a fully registered toolset — eliminating the
    controls cold-start race described in ``primitives._await_mcp_ready``.

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
        model: Model identifier.  When ``None``, resolved from the project's
            ``config.yml`` haiku-tier entry via ``resolve_default_model``.

    Returns:
        SDKWorkflowResult with all collected tool traces, text blocks,
        system messages, MCP server snapshot, and the final ResultMessage.

    Raises:
        ImportError: When ``claude_agent_sdk`` is not installed.
        RuntimeError: When the underlying SDK query fails.
    """
    if not HAS_SDK:
        raise ImportError(
            "claude_agent_sdk is required for run_query. "
            "Install it with: pip install claude-agent-sdk"
        )

    options = build_agent_options(
        project_dir,
        disallowed_tools=disallowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
    )

    workflow = SDKWorkflowResult()

    try:
        # ClaudeSDKClient (streaming) rather than the one-shot ``query()`` so
        # we can poll ``get_mcp_status()`` and wait out async MCP registration
        # before the first turn — eliminating the controls cold-start race.
        async with ClaudeSDKClient(options=options) as client:
            workflow.mcp_servers = await _await_mcp_ready(
                client, _expected_mcp_servers(project_dir)
            )
            await client.query(prompt)
            await _drain_response(client, workflow)
    except Exception as exc:
        raise RuntimeError(f"SDK query failed: {exc}") from exc

    return workflow
