"""OSPREY headless agent-run primitives.

Provides the shared dataclasses, env helpers, and MCP readiness barrier used
by both the ``osprey query`` CLI command and the SDK-based E2E test suite.

Public surface::

    from osprey.agent_runner import (
        SDKWorkflowResult,
        ToolTrace,
        build_agent_options,
        combined_text,
        resolve_default_model,
        sdk_env,
        _expected_mcp_servers,
        _await_mcp_ready,
        run_query,
        agent_session,
        run_turns,
        AgentSession,
        AgentSessionBudgetExceeded,
        TurnResult,
    )
"""

from osprey.agent_runner.primitives import (
    SDKWorkflowResult,
    ToolTrace,
    _await_mcp_ready,
    _expected_mcp_servers,
    build_agent_options,
    combined_text,
    resolve_default_model,
    sdk_env,
)
from osprey.agent_runner.runner import run_query
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

__all__ = [
    # primitives
    "SDKWorkflowResult",
    "ToolTrace",
    "build_agent_options",
    "combined_text",
    "resolve_default_model",
    "sdk_env",
    "_expected_mcp_servers",
    "_await_mcp_ready",
    # runner (single-turn)
    "run_query",
    # session (multi-turn)
    "agent_session",
    "run_turns",
    "AgentSession",
    "AgentSessionBudgetExceeded",
    "TurnResult",
    # write-tool guard
    "load_write_tools",
    "read_only_disallowed_tools",
    # verdict + exit codes
    "evaluate_verdict",
    "EXIT_PASS",
    "EXIT_VERDICT_FAIL",
    "EXIT_USAGE",
]
