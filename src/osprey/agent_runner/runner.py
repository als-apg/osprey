"""Headless agent-run entry point for the ``osprey query`` CLI command.

Provides a production-ready ``run_query`` coroutine built on the shared
primitives in ``osprey.agent_runner.primitives``.  The implementation mirrors
``tests/e2e/sdk_helpers.run_sdk_query`` without the test-only concerns
(e2e budget scaling, subagent transcript harvesting, MCP sidecar persistence).

The module remains importable when ``claude_agent_sdk`` is absent; the runtime
path will raise ``ImportError`` in that case, but module-level imports (e.g.
for type checking or CLI argument parsing) still succeed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient

# SDK imports — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from osprey.agent_runner.primitives import (
    SDKWorkflowResult,
    ToolTrace,
    _await_mcp_ready,
    _expected_mcp_servers,
    _ingest_tool_result,
    _resolve_project_spec,
    resolve_default_model,
    sdk_env,
)
from osprey.infrastructure.proxy.lifecycle import start_proxy

logger = logging.getLogger(__name__)

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
    """Run a read-only agent query via the Claude Agent SDK.

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

    resolved_model = model if model is not None else resolve_default_model(project_dir)

    env = sdk_env(project_dir)

    # Non-native (OpenAI-protocol) providers need the in-process translation
    # proxy: repoint ANTHROPIC_BASE_URL at a loopback proxy that speaks
    # Anthropic to the SDK and the provider's protocol upstream. The proxy is
    # started from spec.upstream_base_url — the OpenAI root *with* its /v1 —
    # NOT from env["ANTHROPIC_BASE_URL"], which the resolver strips of /v1 for
    # Claude Code (see claude_code_resolver); sourcing the upstream from the env
    # var would forward to a /v1-less "…/chat/completions" (issue #312). The
    # proxy auth token lives in the env dict here (provider_env_for_project
    # injected it), not os.environ — see the os.environ-delivery variant in
    # claude_cmd. In production this never fights the e2e proxy override:
    # run_query is production-only; the e2e harness uses run_sdk_query.
    spec = _resolve_project_spec(project_dir)
    if spec and spec.needs_proxy and spec.upstream_base_url:
        auth_token = env.get(spec.auth_env_var)
        if not auth_token:
            # Mirror claude_cmd's pre-flight auth warning: a missing token here
            # otherwise surfaces only as an opaque proxy 401 mid-query.
            logger.warning(
                "Auth token %s missing for provider '%s' — proxied requests may "
                "fail to authenticate (set the provider secret in the project .env)",
                spec.auth_env_var,
                spec.provider,
            )
        port = start_proxy(spec.upstream_base_url, auth_token)
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    options = ClaudeAgentOptions(
        model=resolved_model,
        cwd=str(project_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        env=env,
        setting_sources=["project"],
        disallowed_tools=disallowed_tools,
    )

    workflow = SDKWorkflowResult()

    # Map tool_use_id → ToolTrace for matching results to calls.
    pending_tools: dict[str, ToolTrace] = {}

    try:
        # ClaudeSDKClient (streaming) rather than the one-shot ``query()`` so
        # we can poll ``get_mcp_status()`` and wait out async MCP registration
        # before the first turn — eliminating the controls cold-start race.
        # Message handling is identical to the query() iterator.
        async with ClaudeSDKClient(options=options) as client:
            workflow.mcp_servers = await _await_mcp_ready(
                client, _expected_mcp_servers(project_dir)
            )
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
        raise RuntimeError(f"SDK query failed: {exc}") from exc

    return workflow
