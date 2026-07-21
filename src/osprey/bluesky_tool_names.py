"""Single source of truth for Bluesky MCP tool names.

The Bluesky safety wiring — kill-switch hook matchers, destructive-marker
checks, and the registry permission allow/ask lists — historically string-
matched tool names inline, with no shared vocabulary. Every rename was then a
latent gate-detachment hazard: a tool could be renamed in its module while a
matcher in ``registry/mcp.py`` still pointed at the old name, silently
detaching a safety hook.

This leaf module holds every Bluesky tool name as registered today, so the
registry gate wiring and the destructive-marker classifier can import symbols
instead of repeating string literals. It imports nothing from ``osprey`` and is
safe to import from ``mcp_server``, ``registry``, and ``agent_runner`` code.

Names here reflect the *current* registered surface. Renames are made here
first (changing a single constant), then the consumers follow — keeping every
gate attached across a rename by construction.
"""

# The MCP server short-name under which these tools register. Kill-switch and
# approval hook matchers are built as ``mcp__<server>__<tool>``.
SERVER_NAME = "bluesky"

# --- Read tools -----------------------------------------------------------
# Reach no hardware; auto-approved (registry ``permissions_allow``), no hook.
GET_RUN = "get_run"
LIST_PLANS = "list_plans"
LIST_RUNS = "list_runs"
GET_RUN_DATA = "get_run_data"

# --- Draft tools ----------------------------------------------------------
# Edit the shared plan draft only; touch no hardware (registry
# ``permissions_allow``). ``clear_draft`` matches ``DESTRUCTIVE_MARKERS``
# ("clear") and is blocked under the headless read-only floor by design.
GET_DRAFT = "get_draft"
SET_DRAFT = "set_draft"
CLEAR_DRAFT = "clear_draft"

# --- Authoring tools ------------------------------------------------------
# Write/validate a plan body; carry approval (registry ``permissions_ask``).
WRITE_PLAN = "write_plan"
VALIDATE_PLAN = "validate_plan"

# --- Run-control tools ----------------------------------------------------
# ``launch_run`` starts a real scan (writes-check + approval); ``stop_run`` is
# the safe direction (approval only). Both in registry ``permissions_ask``.
LAUNCH_RUN = "launch_run"
STOP_RUN = "stop_run"

# Every registered Bluesky tool name, grouped as the registry gates them.
READ_TOOLS: tuple[str, ...] = (
    GET_RUN,
    LIST_PLANS,
    LIST_RUNS,
    GET_RUN_DATA,
)
DRAFT_TOOLS: tuple[str, ...] = (
    GET_DRAFT,
    SET_DRAFT,
    CLEAR_DRAFT,
)
AUTHORING_TOOLS: tuple[str, ...] = (
    WRITE_PLAN,
    VALIDATE_PLAN,
)
RUN_CONTROL_TOOLS: tuple[str, ...] = (
    LAUNCH_RUN,
    STOP_RUN,
)
ALL_TOOLS: tuple[str, ...] = (
    *READ_TOOLS,
    *DRAFT_TOOLS,
    *AUTHORING_TOOLS,
    *RUN_CONTROL_TOOLS,
)

# Substrings in a tool name that mark it as destroying stored state. Consumed
# by ``agent_runner.write_tools`` to auto-classify a side-effecting tool that
# sits in an auto-approve list, so it stays blocked under the headless
# read-only floor. Generic across MCP servers, not Bluesky-specific — kept here
# as the safety vocabulary the same wiring depends on.
DESTRUCTIVE_MARKERS: tuple[str, ...] = (
    "delete",
    "remove",
    "clear",
    "wipe",
    "purge",
    "destroy",
)


def matcher(tool_name: str) -> str:
    """Return the ``mcp__<server>__<tool>`` hook-matcher form of a tool name.

    Registry ``HookRule`` matchers and SDK disallow entries address a tool by
    this fully-qualified form; the bare constants above are the short names the
    tool modules register.
    """
    return f"mcp__{SERVER_NAME}__{tool_name}"
