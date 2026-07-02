"""Dispatch permission policy — the trigger allowlist as single authority.

Two per-run enforcement layers for headless dispatch:

* ``make_pretooluse_hook`` builds the **single authority**: a programmatic
  PreToolUse hook (``ClaudeAgentOptions.hooks``). Unlike ``can_use_tool`` —
  which the CLI never consults for calls already permitted by
  ``settings.json`` ``permissions.allow`` rules — a PreToolUse hook fires for
  every tool call, including inside subagents, so neither project settings
  nor other hooks' allow decisions can widen a dispatch run's tool surface.

* ``make_backstop`` builds a context-aware ``can_use_tool`` callback as
  defense-in-depth beneath the hook (e.g. for asks downgraded by concurrent
  settings-hook decisions).

Both are **context-aware** rather than a flat union: the main thread is held
to the trigger's ``allowed_tools``; a subagent is held to its own declared
``tools:`` surface (see ``agent_surfaces.parse_project_agents``). Subagent
context is detected by ``agent_id`` presence — ``agent_type`` alone also
appears on the main thread of ``--agent`` sessions.

The hook is deny-only: an allowed call returns ``{}`` (no decision) so the
facility's own PreToolUse safety hooks and the normal permission flow still
apply. It never emits ``permissionDecision: "allow"``, which would bypass
the permission system (and with it the facility write-approval ask-gate).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from osprey.utils.tool_rules import matches_denylist

try:
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    CLAUDE_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the SDK
    CLAUDE_SDK_AVAILABLE = False
    PermissionResultAllow = object  # type: ignore[assignment,misc]
    PermissionResultDeny = object  # type: ignore[assignment,misc]

logger = logging.getLogger("osprey.mcp_server.dispatch_worker.tool_policy")

# Permission-free harness tools the CLI lets an agent use without any allow
# rule today; a strict deny-only hook would otherwise starve them (TodoWrite
# progress tracking, WaitForMcpServers for MCP cold-start races). Deliberately
# NOT included: Read/Glob/Grep — main-thread file access would expose e.g.
# config.yml provider settings. Denylist entries still beat this set.
PASSTHROUGH_TOOLS = frozenset({"TodoWrite", "WaitForMcpServers"})

# Both names the CLI has used for the subagent-delegation tool.
DELEGATION_TOOLS = ("Task", "Agent")

AgentSurfaces = Mapping[str, "frozenset[str] | None"]


def _deny(reason: str) -> dict[str, Any]:
    """Build a PreToolUse deny decision in the CLI's hook-output shape."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def make_pretooluse_hook(
    trigger_tools: Iterable[str],
    agent_surfaces: AgentSurfaces,
    denied_tools: Iterable[str] = (),
) -> Callable[[dict[str, Any], str | None, Any], Awaitable[dict[str, Any]]]:
    """Build the deny-only PreToolUse hook enforcing the dispatch allowlist.

    Args:
        trigger_tools: The trigger's ``allowed_tools`` — the main thread's
            entire surface.
        agent_surfaces: Declared subagent surfaces from
            ``parse_project_agents`` (``None`` = declared but non-delegable).
        denied_tools: Server denylist (``*`` suffix = prefix match), always
            checked first.

    Returns:
        An async hook callback returning ``{}`` for allowed calls and an
        explicit deny decision otherwise. Internal errors deny (fail-closed).
    """
    trigger_set = frozenset(trigger_tools)
    denied_tuple = tuple(denied_tools)

    async def pretooluse_hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        try:
            tool_name = str(input_data.get("tool_name") or "")

            if matches_denylist(tool_name, denied_tuple):
                return _deny(f"Tool {tool_name!r} is blocked by the dispatch server denylist")

            if tool_name in PASSTHROUGH_TOOLS:
                return {}

            if tool_name in DELEGATION_TOOLS:
                tool_input = input_data.get("tool_input") or {}
                target = tool_input.get("subagent_type")
                if not target:
                    return _deny(
                        "delegation requires an explicit subagent_type; "
                        "general-purpose is not permitted in dispatch"
                    )
                if target not in agent_surfaces:
                    return _deny(
                        f"agent {target!r} is not declared in this project's .claude/agents/"
                    )
                if agent_surfaces[target] is None:
                    return _deny(
                        f"agent {target!r} has no explicit tools: list in its "
                        ".claude/agents/ file; declare one to enable it for dispatch"
                    )
                return {}

            # Subagent context iff agent_id is present (agent_type alone also
            # appears on --agent main threads).
            if "agent_id" in input_data:
                agent_type = str(input_data.get("agent_type") or "")
                surface = agent_surfaces.get(agent_type)
                if surface is None:
                    return _deny(
                        f"Tool {tool_name!r} denied: unknown or tools-less subagent "
                        f"context {agent_type!r}"
                    )
                if tool_name in surface:
                    return {}
                return _deny(
                    f"Tool {tool_name!r} is not in subagent {agent_type!r}'s declared tools list"
                )

            if tool_name in trigger_set:
                return {}
            return _deny(f"Tool {tool_name!r} is not in this trigger's allowed_tools list")
        except Exception:
            logger.exception("dispatch permission hook internal error — denying tool")
            return _deny("dispatch permission hook internal error (fail-closed)")

    return pretooluse_hook


def make_backstop(
    trigger_tools: Iterable[str],
    agent_surfaces: AgentSurfaces,
    denied_tools: Iterable[str] = (),
) -> Any:
    """Build the context-aware ``can_use_tool`` backstop under the hook.

    Main thread (``context.agent_id`` is None) is held to the trigger set;
    subagent context to the union of all declared surfaces —
    ``ToolPermissionContext`` carries no ``agent_type``, so per-agent
    precision is the hook's job; the union merely never widens the main
    thread (which today's flat allowlist would).
    """
    trigger_set = frozenset(trigger_tools)
    denied_tuple = tuple(denied_tools)
    subagent_union = frozenset().union(
        *(surface for surface in agent_surfaces.values() if surface is not None)
    )

    async def can_use_tool(tool_name, tool_input, context):  # type: ignore[no-untyped-def]
        if matches_denylist(tool_name, denied_tuple):
            return PermissionResultDeny(
                message=f"Tool {tool_name!r} is blocked by the dispatch server denylist",
            )
        if tool_name in PASSTHROUGH_TOOLS:
            return PermissionResultAllow()
        allowed = trigger_set if getattr(context, "agent_id", None) is None else subagent_union
        if tool_name in allowed:
            return PermissionResultAllow()
        return PermissionResultDeny(
            message=f"Tool {tool_name!r} is not permitted in this dispatch context",
        )

    return can_use_tool
