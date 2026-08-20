"""Single source of truth for Bluesky MCP tool names.

The Bluesky safety wiring — kill-switch hook matchers, destructive-marker
checks, and the registry permission allow/ask lists — must never string-match
tool names inline. Inline literals make every rename a latent gate-detachment
hazard: a tool renamed in its module while a matcher in ``registry/mcp.py``
still names the previous string silently detaches a safety hook.

This leaf module holds every Bluesky tool name as registered, so the registry
gate wiring and the destructive-marker classifier can import symbols instead of
repeating string literals. It imports nothing from ``osprey`` and is safe to
import from ``mcp_server``, ``registry``, and ``agent_runner`` code.

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
LIST_DEVICES = "list_devices"
LIST_RUNS = "list_runs"
GET_RUN_DATA = "get_run_data"
GET_RUN_FIGURE = "get_run_figure"
GET_PLAN_SOURCE = "get_plan_source"

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

# --- Queue tools ----------------------------------------------------------
# Execution is two steps: ``queue_add`` puts the pinned draft in the queue,
# ``queue_start`` drains it. Both arm hardware motion (writes-check +
# approval). ``queue_stop`` carries approval only — a plain stop is the safe
# direction and must never be kill-switch-blocked, and its one arming case
# (withdrawing a pending stop) is gated in-tool and at the bridge instead, so
# that halting keeps working when the kill switch is on. ``queue_list`` and
# ``queue_status`` are reads (registry ``permissions_allow``).
QUEUE_LIST = "queue_list"
QUEUE_STATUS = "queue_status"
QUEUE_ADD = "queue_add"
QUEUE_START = "queue_start"
QUEUE_STOP = "queue_stop"

# --- Run-control tools ----------------------------------------------------
# ``stop_run`` is the emergency abort: it stops the plan already in motion
# (POST /queue/abort), where ``queue_stop`` only halts the queue after the
# running item finishes. Like ``queue_stop`` it is the safe direction and
# carries approval ONLY — never ``_WRITES_CHECK`` — so the kill switch can
# never block a halt; registry ``permissions_ask``.
STOP_RUN = "stop_run"

# Every registered Bluesky tool name, grouped as the registry gates them.
READ_TOOLS: tuple[str, ...] = (
    GET_RUN,
    LIST_PLANS,
    LIST_DEVICES,
    LIST_RUNS,
    GET_RUN_DATA,
    GET_RUN_FIGURE,
    GET_PLAN_SOURCE,
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
QUEUE_READ_TOOLS: tuple[str, ...] = (
    QUEUE_LIST,
    QUEUE_STATUS,
)
QUEUE_CONTROL_TOOLS: tuple[str, ...] = (
    QUEUE_ADD,
    QUEUE_START,
    QUEUE_STOP,
)
QUEUE_TOOLS: tuple[str, ...] = (
    *QUEUE_READ_TOOLS,
    *QUEUE_CONTROL_TOOLS,
)
RUN_CONTROL_TOOLS: tuple[str, ...] = (STOP_RUN,)

# The tools that arm hardware motion, and so must carry the kill-switch hook
# (registry ``_WRITES_CHECK``) on top of the approval prompt. Named as its own
# group because "which tools are writes-gated" is a safety claim worth
# asserting directly rather than re-deriving from a hook list: a tool added to
# the queue group but omitted here would register with an approval prompt and
# no kill switch, which looks gated and is not.
ARMING_TOOLS: tuple[str, ...] = (
    QUEUE_ADD,
    QUEUE_START,
)
ALL_TOOLS: tuple[str, ...] = (
    *READ_TOOLS,
    *DRAFT_TOOLS,
    *AUTHORING_TOOLS,
    *QUEUE_TOOLS,
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
