"""OSPREY headless agent-run primitives.

The Claude adapter: option building, provider routing, the MCP readiness
barrier, single-prompt and multi-turn runs, and the plain event records that
report agent output. A caller outside this package reads those records and
never handles an agent SDK type.

Importing the package, or any one module of it, costs only that module. The
build and deploy layers read the tool-name lists and path helpers here, and
they must not load the agent SDK to do it, so every re-exported name resolves
on first access.

Public surface::

    from osprey.agent_runner import (
        SDKWorkflowResult,
        ToolTrace,
        build_agent_options,
        combined_text,
        resolve_default_model,
        sdk_env,
        expected_mcp_servers,
        await_mcp_ready,
        run_query,
        stream_query,
        agent_session,
        run_turns,
        AgentSession,
        AgentSessionBudgetExceeded,
        TurnResult,
        AgentEvent,
        TextEvent,
        ThinkingEvent,
        ToolUseEvent,
        ToolResultEvent,
        ApiErrorEvent,
        SystemEvent,
        ResultEvent,
        AgentRunError,
        McpNotReadyError,
        HAS_SDK,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.agent_runner.errors import AgentRunError, McpNotReadyError
    from osprey.agent_runner.events import (
        AgentEvent,
        ApiErrorEvent,
        ResultEvent,
        SystemEvent,
        TextEvent,
        ThinkingEvent,
        ToolResultEvent,
        ToolUseEvent,
    )
    from osprey.agent_runner.primitives import (
        HAS_SDK,
        MCP_READY_TIMEOUT_S,
        SDKWorkflowResult,
        ToolTrace,
        await_mcp_ready,
        build_agent_options,
        combined_text,
        expected_mcp_servers,
        mcp_servers_connected,
        mcp_snapshot_summary,
        resolve_default_model,
        sdk_env,
    )
    from osprey.agent_runner.runner import run_query, stream_query
    from osprey.agent_runner.session import (
        AgentSession,
        AgentSessionBudgetExceeded,
        TurnResult,
        agent_session,
        run_turns,
    )
    from osprey.agent_runner.verdict import (
        EXIT_PASS,
        EXIT_USAGE,
        EXIT_VERDICT_FAIL,
        evaluate_verdict,
    )
    from osprey.agent_runner.write_tools import load_write_tools, read_only_disallowed_tools

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "AgentRunError": ".errors",
    "McpNotReadyError": ".errors",
    "AgentEvent": ".events",
    "ApiErrorEvent": ".events",
    "ResultEvent": ".events",
    "SystemEvent": ".events",
    "TextEvent": ".events",
    "ThinkingEvent": ".events",
    "ToolResultEvent": ".events",
    "ToolUseEvent": ".events",
    "HAS_SDK": ".primitives",
    "MCP_READY_TIMEOUT_S": ".primitives",
    "SDKWorkflowResult": ".primitives",
    "ToolTrace": ".primitives",
    "await_mcp_ready": ".primitives",
    "build_agent_options": ".primitives",
    "combined_text": ".primitives",
    "expected_mcp_servers": ".primitives",
    "mcp_servers_connected": ".primitives",
    "mcp_snapshot_summary": ".primitives",
    "resolve_default_model": ".primitives",
    "sdk_env": ".primitives",
    "run_query": ".runner",
    "stream_query": ".runner",
    "AgentSession": ".session",
    "AgentSessionBudgetExceeded": ".session",
    "TurnResult": ".session",
    "agent_session": ".session",
    "run_turns": ".session",
    "EXIT_PASS": ".verdict",
    "EXIT_USAGE": ".verdict",
    "EXIT_VERDICT_FAIL": ".verdict",
    "evaluate_verdict": ".verdict",
    "load_write_tools": ".write_tools",
    "read_only_disallowed_tools": ".write_tools",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
