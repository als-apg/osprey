"""Unified server and agent registry for Claude Code integration.

Replaces scattered Jinja2 template guards with a data-driven registry.
Templates become generic loops; all server/agent metadata lives here.

Users extend Osprey through ``config.yml`` — no framework source changes needed.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from osprey import bluesky_tool_names as bsky
from osprey.audit.posture import POSTURE_ENV_VAR
from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.utils.identity import AUDIT_IDENTITY_ENV as AUDIT_IDENTITY_ENV  # re-exported
from osprey.utils.identity import IDENTITY_ENV_LADDER
from osprey.utils.workspace import RENDERED_CONFIG_RELPATH

logger = logging.getLogger(__name__)

#: ``OSPREY_CONFIG`` / ``CONFIG_FILE`` value for every framework server, as an
#: unresolved template — :func:`_resolve_placeholder` substitutes
#: ``{project_root}`` at render time. The env vars themselves are the runtime
#: contract every server reads and are deliberately untouched; only the path
#: they carry moved, from the repo root into the render zone beside it.
RENDERED_CONFIG_ENV_VALUE = f"{{project_root}}/{RENDERED_CONFIG_RELPATH}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class HookEntry:
    """A single hook command."""

    type: str = "command"
    command: str = ""
    timeout: int = 5


@dataclass
class HookRule:
    """A matcher + list of hooks for PreToolUse / PostToolUse."""

    matcher: str
    hooks: list[HookEntry] = field(default_factory=list)


@dataclass
class ServerDefinition:
    """Metadata for one MCP server."""

    name: str
    module: str  # e.g. "osprey.mcp_server.control_system"
    env: dict[str, str] = field(default_factory=dict)
    args_extra: list[str] = field(default_factory=list)  # extra args after ["-m", module]
    condition: str | None = None  # ctx key that must be truthy
    default_enabled: bool = True
    permissions_allow: list[str] = field(default_factory=list)
    permissions_ask: list[str] = field(default_factory=list)
    fixed_allow: list[str] = field(default_factory=list)
    fixed_ask: list[str] = field(default_factory=list)
    hooks_pre: list[HookRule] = field(default_factory=list)
    hooks_post: list[HookRule] = field(default_factory=list)
    is_external: bool = False
    external_command: str | None = None
    external_args: list[str] = field(default_factory=list)
    url: str | None = None  # Remote transport URL (mutually exclusive with command)
    # Wire transport for URL servers: "http" (streamable-HTTP) or "sse"
    # (legacy Server-Sent Events). Meaningless for stdio servers.
    transport: str = "http"
    port: int | None = (
        None  # Host/container port for HTTP servers; informational for non-Claude consumers
    )
    # Framework template name this definition was cloned from via ``extends``
    # (set by build_extended_server); None for framework/custom definitions.
    extends_of: str | None = None


# ---------------------------------------------------------------------------
# Hook helpers (reduce repetition in FRAMEWORK_SERVERS)
# ---------------------------------------------------------------------------

_APPROVAL = HookEntry(
    command='python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_approval.py"',
    timeout=5,
)
_WRITES_CHECK = HookEntry(
    command='python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_writes_check.py"',
    timeout=5,
)
_LIMITS = HookEntry(
    command='python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_limits.py"',
    timeout=10,
)
_ERROR_GUIDANCE = HookEntry(
    command='python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_error_guidance.py"',
    timeout=5,
)
_CF_FEEDBACK = HookEntry(
    command='python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_cf_feedback_capture.py"',
    timeout=10,
)


HOOK_PRESETS: dict[str, HookEntry] = {
    "approval": _APPROVAL,
    "writes_check": _WRITES_CHECK,
    "limits": _LIMITS,
}


def _post_error(matcher: str) -> HookRule:
    """Standard PostToolUse error-guidance hook for a server."""
    return HookRule(matcher=matcher, hooks=[_ERROR_GUIDANCE])


# ---------------------------------------------------------------------------
# Framework server catalog
# ---------------------------------------------------------------------------

FRAMEWORK_SERVERS: dict[str, ServerDefinition] = {
    "controls": ServerDefinition(
        name="controls",
        module="osprey.mcp_server.control_system",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
            "EPICS_CA_ADDR_LIST": "${EPICS_CA_ADDR_LIST:-}",
        },
        # control_target only reports: it derives endpoints from config, reads
        # the prober's cache and asks the manager what it already knows. No
        # socket, no child, no state write — so prompting for it would train
        # operators to click through a prompt that never precedes motion.
        permissions_allow=["channel_limits", "control_target"],
        # control_target_set moves the whole session to another machine, so it
        # is approval-gated — but deliberately NOT writes-check gated. The
        # writes kill switch renders pure-write tools into permissions.deny,
        # and a deployment with writes off is exactly the one that most needs to
        # be able to move a session between the simulator and the machine it
        # only reads. Its own refusals (read-only run, execution in flight,
        # eligibility) are the gate that matters here.
        permissions_ask=["channel_write", "control_target_set"],
        hooks_pre=[
            HookRule(
                matcher="mcp__controls__channel_write",
                hooks=[_WRITES_CHECK, _LIMITS, _APPROVAL],
            ),
            HookRule(
                matcher="mcp__controls__control_target_set",
                hooks=[_APPROVAL],
            ),
            HookRule(
                matcher="mcp__controls__channel_read",
                hooks=[_APPROVAL],
            ),
            HookRule(
                matcher="mcp__controls__archiver_read",
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__controls__.*")],
    ),
    "phoebus": ServerDefinition(
        name="phoebus",
        module="osprey.mcp_server.phoebus",
        # Off by default: only profiles that opt in (claude_code.servers.phoebus.enabled
        # = true) get the native-Phoebus tools. A second live instance is declared in
        # config as claude_code.servers.<name>.extends: phoebus — see build_extended_server().
        default_enabled=False,
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
            # Full-URL override of the in-JVM bridge (default 127.0.0.1:7979).
            "PHOEBUS_BRIDGE_URL": "${PHOEBUS_BRIDGE_URL:-http://127.0.0.1:7979}",
            # Instance identity — tools tag UI signals with it (open_panel →
            # panel_focus targets the matching web-terminal tab). extends
            # clones get this auto-rewritten to their own name.
            "OSPREY_SERVER_NAME": "phoebus",
        },
        permissions_allow=[
            "phoebus_list_displays",
            "phoebus_perceive",
            "phoebus_perceive_region",
            "phoebus_snapshot",
            # Opening a display or a Data Browser plot touches no PVs and
            # actuates nothing — allow.
            "phoebus_open_panel",
            "phoebus_open_databrowser",
        ],
        # Driving a live panel actuates hardware-facing controls — gate on approval.
        permissions_ask=["phoebus_drive"],
        hooks_pre=[
            HookRule(
                matcher="mcp__phoebus__phoebus_drive",
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__phoebus__.*")],
    ),
    "python": ServerDefinition(
        name="python",
        module="osprey.mcp_server.python_executor",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        },
        permissions_allow=[],
        permissions_ask=["execute"],
        hooks_pre=[
            HookRule(
                matcher="mcp__python__execute",
                hooks=[_WRITES_CHECK, _APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__python__.*")],
    ),
    "osprey_workspace": ServerDefinition(
        name="osprey_workspace",
        module="osprey.mcp_server.workspace",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            # osprey.utils.config reads CONFIG_FILE (not OSPREY_CONFIG); set both
            # so the server resolves config even when launched with a CWD other
            # than the project dir (e.g. the dispatch worker's /app WORKDIR).
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        },
        permissions_allow=[
            "facility_description",
            "screenshot_capture",
            "list_windows",
            "manage_window",
            "submit_response",
            "artifact_list",
            "artifact_read",
            "artifact_get",
            "artifact_focus",
            "artifact_export",
            "create_static_plot",
            "create_interactive_plot",
            "create_dashboard",
            "create_document",
            "artifact_register",
            "artifact_delete",
            "artifact_delete_all",
            "provenance_locator",
            "session_log",
            "session_summary",
            "archiver_downsample",
            "setup_inspect",
            "lattice_init",
            "lattice_state",
            "lattice_set_param",
            "lattice_refresh",
            "lattice_set_baseline",
            "list_panels",
            # The on-screen axis: both halves are reversible in one operator
            # click, so neither is worth a prompt. The rail axis
            # (add_panel_to_rail/remove_panel_from_rail) is deliberately absent —
            # taking a panel off the rail costs the operator the ability to
            # launch it back, which is worth asking about.
            "open_panel",
            "close_panel",
            "arrange_workspace",
        ],
        permissions_ask=["setup_patch"],
        hooks_pre=[
            HookRule(
                matcher="mcp__osprey_workspace__setup_patch",
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__osprey_workspace__.*")],
    ),
    "ariel": ServerDefinition(
        name="ariel",
        module="osprey.mcp_server.ariel",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
        },
        permissions_allow=[
            "keyword_search",
            "semantic_search",
            "hybrid_search",
            "sql_query",
            "entries_by_ids",
            "browse",
            "entry_get",
            "capabilities",
            "status",
            "filter_options",
        ],
        permissions_ask=["entry_create"],
        hooks_pre=[
            HookRule(
                matcher="mcp__ariel__entry_create",
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__ariel__.*")],
    ),
    "osprey_facility_knowledge": ServerDefinition(
        name="osprey_facility_knowledge",
        module="osprey.mcp_server.facility_knowledge",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
        },
        permissions_allow=["capabilities", "list_concepts", "read_concept", "search"],
        permissions_ask=["draft_concept"],
        hooks_pre=[
            HookRule(
                matcher="mcp__osprey_facility_knowledge__draft_concept",
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__osprey_facility_knowledge__.*")],
    ),
    "bluesky": ServerDefinition(
        name="bluesky",
        module="osprey.mcp_server.bluesky",
        # Off by default: only profiles that opt in (claude_code.servers.bluesky.enabled
        # = true) get the Bluesky bridge client tools — running them requires a live
        # facility-side Bluesky bridge process (mirrors phoebus's opt-in reasoning).
        default_enabled=False,
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
            # No literal default: an unset variable reaches the server EMPTY,
            # which resolve_bridge_url reads as "no override" and falls back
            # to the rendered config — `services.bluesky.port`, the port this
            # deployment actually publishes. A `:-http://127.0.0.1:8090`
            # default here was a second copy of the bridge's port that the
            # config could never correct, so a bridge moved on the profile
            # left every agent dialing the old port.
            "BLUESKY_BRIDGE_URL": "${BLUESKY_BRIDGE_URL:-}",
            "BLUESKY_LAUNCH_TOKEN": "${BLUESKY_LAUNCH_TOKEN:-}",
        },
        # Tool names resolve from osprey.bluesky_tool_names (the single source of
        # truth) so a rename there follows through every gate here by construction.
        permissions_allow=[
            bsky.GET_RUN,
            bsky.LIST_PLANS,
            bsky.LIST_DEVICES,
            bsky.LIST_RUNS,
            bsky.GET_RUN_DATA,
            bsky.GET_RUN_FIGURE,
            bsky.GET_PLAN_SOURCE,
            # Draft tools never touch hardware — editing the shared
            # plan draft only stages what a future queue_add or in-panel
            # Add-to-queue click might queue, so like the read tools above they need no approval
            # prompt and carry no _WRITES_CHECK hook. clear_draft is
            # nonetheless auto-classified side-effecting by
            # agent_runner.write_tools (matches bsky.DESTRUCTIVE_MARKERS'
            # "clear") and blocked under the headless read-only floor
            # regardless of this allow-listing — acceptable, expected
            # posture; do not rename the tool to dodge it.
            bsky.GET_DRAFT,
            bsky.SET_DRAFT,
            bsky.CLEAR_DRAFT,
            # Queue reads: queue_list reads the queue the manager
            # holds, queue_status reads whether this deployment can execute at
            # all. Neither mutates anything, and queue_status in particular is
            # what an agent should call BEFORE composing a plan — putting an
            # approval prompt on that question would train operators to click
            # through prompts that never precede motion.
            *bsky.QUEUE_READ_TOOLS,
        ],
        # queue_add and queue_start are the arming pair (bsky.ARMING_TOOLS):
        # adding hands an item to a queue that may already be draining, and
        # starting drains it — both carry _WRITES_CHECK plus approval.
        # queue_stop (halt the queue after the running item) and stop_run
        # (abort the plan already in motion) are the safe direction and must
        # never be kill-switch-blocked, so they carry approval only; queue_stop's
        # one arming case (cancel=true, which withdraws a pending halt) is gated
        # in-tool and again at the bridge, because attaching _WRITES_CHECK here
        # would also block a PLAIN stop whenever writes are disabled — exactly
        # when halting matters most. stop_run has no arming case at all: it is
        # ungated end to end, at the tool and at the bridge.
        # write_plan/validate_plan reach NO hardware
        # either way: write_plan only writes a file (never imports/execs
        # it), and validate_plan's dry run drives mock devices only, in a
        # subprocess with EPICS_CA_* neutralized — both work identically whether
        # control_system.writes_enabled is on or off, so like stop_run neither
        # carries _WRITES_CHECK. They get their own (distinct, independently
        # allowlistable) short-names rather than reusing the queue tools'
        # tier, since an operator may want to permit authoring/validating plan
        # bodies without also auto-approving queue_add/queue_start, or vice versa.
        permissions_ask=[
            *bsky.QUEUE_CONTROL_TOOLS,
            bsky.STOP_RUN,
            bsky.WRITE_PLAN,
            bsky.VALIDATE_PLAN,
        ],
        hooks_pre=[
            *(
                HookRule(matcher=bsky.matcher(tool), hooks=[_WRITES_CHECK, _APPROVAL])
                for tool in bsky.ARMING_TOOLS
            ),
            HookRule(
                matcher=bsky.matcher(bsky.QUEUE_STOP),
                hooks=[_APPROVAL],
            ),
            HookRule(
                matcher=bsky.matcher(bsky.STOP_RUN),
                hooks=[_APPROVAL],
            ),
            HookRule(
                matcher=bsky.matcher(bsky.WRITE_PLAN),
                hooks=[_APPROVAL],
            ),
            HookRule(
                matcher=bsky.matcher(bsky.VALIDATE_PLAN),
                hooks=[_APPROVAL],
            ),
        ],
        hooks_post=[_post_error("mcp__bluesky__.*")],
    ),
    "health": ServerDefinition(
        name="health",
        module="osprey.mcp_server.health",
        # Off by default: only profiles that opt in (claude_code.servers.health.enabled
        # = true) get the system-health tools (mirrors phoebus's opt-in reasoning).
        #
        # Read-only posture — no _WRITES_CHECK / approval hooks. The health tools take
        # ONLY a categories filter; they accept no channel, URL, or probe parameters, so
        # every connector touch is config-declared and read-only. That is why the
        # auto-approved health_check may execute channel_read probes that would be
        # approval-gated as free-parameter reads on the controls server: here the operator
        # cannot steer the read at call time, so there is nothing to gate.
        default_enabled=False,
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        },
        permissions_allow=["health_check"],
        permissions_ask=["health_check_full"],
        hooks_post=[_post_error("mcp__health__.*")],
    ),
    "channel-finder": ServerDefinition(
        name="channel-finder",
        module="osprey.mcp_server.channel_finder_{channel_finder_pipeline}",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        },
        condition="channel_finder_pipeline",
        # permissions_allow is populated dynamically from
        # CHANNEL_FINDER_TOOLS_BY_PIPELINE in resolve_servers() because the tool
        # set varies by pipeline.
        hooks_post=[
            HookRule(
                matcher="mcp__channel-finder__.*",
                hooks=[_ERROR_GUIDANCE, _CF_FEEDBACK],
            ),
        ],
    ),
    "graph": ServerDefinition(
        name="graph",
        module="osprey.mcp_server.graph",
        env={
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        },
        # Conditional on a declared graph store: ``graphdb_configured`` is a
        # config-derived context key that is truthy only when the project
        # configures ``services.graphdb``. Without the store there is nothing to
        # query, so the server is left out of the render entirely rather than
        # shipped as a tool that can only fail.
        condition="graphdb_configured",
        # Read-only knowledge-graph search — every tool reads and nothing the
        # agent passes at call time can mutate the store, so there is no
        # approval / writes-check hook here and permissions_ask stays empty:
        # read_cypher runs inside a read-mode transaction that the store itself
        # rejects writes in, and its Cypher gate refuses extension procedures /
        # functions and LOAD CSV before dialing; get_schema and example_queries
        # serve metadata, and capabilities is a static manifest.
        permissions_allow=["capabilities", "example_queries", "get_schema", "read_cypher"],
        permissions_ask=[],
        hooks_post=[_post_error("mcp__graph__.*")],
    ),
}


# Tools exposed by each channel-finder pipeline (one MCP server module per pipeline).
# The agent template and the server's permissions.allow are both rendered from
# this single source of truth at build time.
CHANNEL_FINDER_TOOLS_BY_PIPELINE: dict[str, list[str]] = {
    "hierarchical": ["build_channels", "get_options", "view_examples"],
    "middle_layer": [
        "get_common_names",
        "inspect_fields",
        "list_channels",
        "list_families",
        "list_systems",
        "run_sql",
        "statistics",
        "validate",
    ],
    "in_context": ["ask_channels"],
    # Same four tools as the standalone ``graph`` server above, and for the same
    # reason: both front the one read-only knowledge-graph vocabulary. They differ
    # in who calls them — the standalone server answers the main agent's facility
    # questions, this pipeline puts the same tools behind the channel-finder
    # subagent so a graph store can serve address lookups the way a channel
    # database does for the file-backed paradigms.
    "graph": ["capabilities", "example_queries", "get_schema", "read_cypher"],
}

# Every keyed pipeline must be a paradigm the registry knows about, or the
# rendered agent would advertise tools for a mode nothing else in the build
# accepts. Checked at import so a typo fails loudly instead of resolving to an
# empty tool list. The reverse is not required: a registered paradigm may be
# served by a pipeline that exposes no MCP tools of its own.
_unregistered_pipelines = set(CHANNEL_FINDER_TOOLS_BY_PIPELINE) - set(VALID_CHANNEL_FINDER_MODES)
if _unregistered_pipelines:
    raise RuntimeError(
        "CHANNEL_FINDER_TOOLS_BY_PIPELINE names channel-finder paradigms that are "
        f"not registered in VALID_CHANNEL_FINDER_MODES: {sorted(_unregistered_pipelines)!r}"
    )
del _unregistered_pipelines


# ---------------------------------------------------------------------------
# Read/write-mixed tools (the writes kill switch's documented exception)
# ---------------------------------------------------------------------------

#: Framework templates whose ``_WRITES_CHECK``-gated tools are read/write
#: MIXED rather than pure-write.
#:
#: ``python``'s ``execute`` is the one documented exception to the writes
#: kill-switch's hard-deny default: it accepts both read-only and
#: write-access kernels, so a readonly posture must keep it reachable and let
#: the writes-check hook (and, server-side, the audit middleware) decide per
#: call. Every other ``_WRITES_CHECK``-gated tool is presumed pure-write and
#: is denied outright when writes are off.
#:
#: It lives in the registry rather than in a renderer because the registry is
#: where ``_WRITES_CHECK`` and ``FRAMEWORK_SERVERS`` are: every consumer — the
#: Claude Code kill switch, the rendered hook config, the MCP audit
#: middleware's clamp set and its degraded-render floor — reads the same
#: classification here instead of restating a set of template names.
MIXED_READ_WRITE_TEMPLATES: frozenset[str] = frozenset({"python"})


def writes_check_matchers(template: str, server_name: str | None = None) -> list[str]:
    """Fully-qualified ``_WRITES_CHECK``-gated tool matchers of one framework template.

    This is the canonical construction every write-gating consumer shares:
    walk the template's ``hooks_pre`` and keep the rules whose hooks include
    the :data:`_WRITES_CHECK` singleton. Identity on the shared ``HookEntry``,
    not a substring search on its command line, so a moved hook path cannot
    quietly stop matching.

    Args:
        template: Name of a :data:`FRAMEWORK_SERVERS` entry. A name the
            registry does not know (a custom, profile-authored server) has no
            template to read rules from and yields ``[]`` rather than raising:
            callers walk mixed server lists that may legitimately contain one.
        server_name: The rendered server's own name, when it is an ``extends``
            clone. The matchers are re-anchored ``mcp__<template>__`` →
            ``mcp__<server_name>__``, the same splice
            :func:`build_extended_server` applies to the clone's hook rules, so
            a clone's tools are named for the clone.

    Returns:
        Matchers in the template's own hook order.
    """
    template_def = FRAMEWORK_SERVERS.get(template)
    if template_def is None:
        return []
    name = server_name or template
    old_prefix, new_prefix = f"mcp__{template}__", f"mcp__{name}__"
    matchers = []
    for rule in template_def.hooks_pre:
        if _WRITES_CHECK not in rule.hooks:
            continue
        matcher = rule.matcher
        if name != template and matcher.startswith(old_prefix):
            matcher = new_prefix + matcher[len(old_prefix) :]
        matchers.append(matcher)
    return matchers


def framework_mixed_read_write_tools() -> list[str]:
    """Every mixed read/write tool the framework itself ships, fully qualified.

    The template-name floor: no project config, no ``extends`` clones. This is
    what a consumer falls back to when it cannot read a render's own list (a
    degraded or missing hook config) — a floor that is never *wider* than the
    real render, because a clone can only add tools to it.

    Sorted by template so the value is stable across interpreter runs
    (``MIXED_READ_WRITE_TEMPLATES`` is a set); within a template, hook order.
    """
    return [
        matcher
        for template in sorted(MIXED_READ_WRITE_TEMPLATES)
        for matcher in writes_check_matchers(template)
    ]


def framework_write_tools() -> list[str]:
    """Every ``_WRITES_CHECK``-gated tool the framework itself ships, fully qualified.

    The whole write-gated set, where :func:`framework_mixed_read_write_tools`
    is its read/write-mixed subset: the matchers of *every*
    :data:`FRAMEWORK_SERVERS` entry that gates a tool on :data:`_WRITES_CHECK`,
    enabled-by-default or not. Template names only — no project config, no
    ``extends`` clones — so it is never *wider* than a real render.

    It exists because the degraded write floor is a drift seam. Two consumers
    carry a hardcoded copy of this list, clamped when a render cannot be read:
    the MCP audit middleware's ``_FALLBACK_WRITE_TOOLS`` and the same-named
    literal in the ``osprey_writes_check.py`` hook. Neither may import the
    registry — one is the running server, which should not import the
    render/launch side to learn what it refuses; the other ships as a
    copied-in template file run by a bare ``python3``. Pinning them only
    against *each other* (which is all they had) makes two identical copies
    agree while both drift away from the registry, which is exactly how
    ``bluesky``'s arming pair came to be missing from the floor. This function
    is the third, derived answer both are pinned against in
    ``tests/registry/test_mixed_floor_driftguard.py``.

    Sorted by template so the value is stable across interpreter runs; within
    a template, hook order.
    """
    return [
        matcher
        for template in sorted(FRAMEWORK_SERVERS)
        for matcher in writes_check_matchers(template)
    ]


def mixed_read_write_tools(servers: list[dict]) -> list[str]:
    """The mixed read/write tools of one render, fully qualified.

    Computed from resolved servers rather than from template names alone
    because that is the only place both halves are known: which mixed servers
    this project actually enables, and what an ``extends`` clone's tools are
    called after the registry rewrote their prefixes. Consumers downstream
    (the rendered hook config, and through it the audit middleware) therefore
    receive a finished list and classify nothing themselves.

    Args:
        servers: Output of :func:`resolve_servers`. Disabled servers are
            skipped — a server that does not render has no tools to exempt —
            and so are custom servers, which name no framework template and
            carry no framework write-gating.

    Returns:
        Deduplicated, in resolved-server order. Rendered into a JSON safety
        file, so a stable order keeps every built project's diff quiet.
    """
    tools: list[str] = []
    for server in servers:
        if not server.get("enabled"):
            continue
        template = server.get("extends_of") or server["name"]
        if template not in MIXED_READ_WRITE_TEMPLATES:
            continue
        for matcher in writes_check_matchers(template, server["name"]):
            if matcher not in tools:
                tools.append(matcher)
    return tools


# ---------------------------------------------------------------------------
# Agent data model and catalog
# ---------------------------------------------------------------------------


@dataclass
class AgentDefinition:
    """Metadata for one Claude Code agent."""

    name: str
    template_path: str | None = None
    condition: str | None = None
    server_dependency: str | None = None
    default_enabled: bool = True
    description: str = ""
    is_custom: bool = False
    # Approval-gated (permissions.ask) tools this agent hard-requires. These
    # must survive the writes-disabled kill-switch that otherwise pulls
    # read/write tools like mcp__python__execute out of ask (see
    # cli/templates/claude_code.py) -- an enabled agent declaring a tool that
    # is neither in allow nor ask fails build validation.
    requires_ask_tools: frozenset[str] = frozenset()


FRAMEWORK_AGENTS: dict[str, AgentDefinition] = {
    "channel-finder": AgentDefinition(
        name="channel-finder",
        condition="channel_finder_pipeline",
        description="Finds channel/PV addresses. You do NOT have channel-finding tools.",
    ),
    "logbook-search": AgentDefinition(
        name="logbook-search",
        description="Searches the facility logbook for entries and events.",
    ),
    "logbook-deep-research": AgentDefinition(
        name="logbook-deep-research",
        description="Complex multi-step logbook investigations.",
    ),
    "data-visualizer": AgentDefinition(
        name="data-visualizer",
        server_dependency="python",
        description=(
            "Creates plots, dashboards, and compiles LaTeX documents. "
            "You do NOT have visualization tools."
        ),
    ),
    "facility-knowledge": AgentDefinition(
        name="facility-knowledge",
        description=(
            "Answers questions about facility design, accelerator physics concepts, "
            "and operational knowledge from the facility knowledge bundle. Delegate to "
            "this agent when the user asks about facility layout, terminology, beam "
            "parameters, or any documented facility knowledge."
        ),
    ),
    "facility-knowledge-graph": AgentDefinition(
        name="facility-knowledge-graph",
        # Rides the graph server, not just the store: an explicit
        # ``claude_code.servers.graph.enabled: false`` takes the agent away
        # together with the tools it delegates to.
        server_dependency="graph",
        description=(
            "Answers structural questions about the machine from the facility "
            "knowledge graph: which devices exist, where they sit along the beam, "
            "what channels a device exposes, which addresses read vs write, device "
            "classes and counts. Delegate to this agent when the user asks how the "
            "machine fits together."
        ),
    ),
    "pyat-specialist": AgentDefinition(
        name="pyat-specialist",
        server_dependency="python",
        description=(
            "Delegate to this agent when the user needs lattice/optics quantities "
            "computed from the accelerator model (orbit, tunes, beta functions, "
            "dispersion, response matrices) — it writes and executes pyAT code "
            "against the simulated ALS-U AR ring."
        ),
        # Its only compute path is mcp__python__execute (read-only kernels), so
        # it must keep that tool even in a writes-disabled (read-only) persona.
        requires_ask_tools=frozenset({"mcp__python__execute"}),
    ),
}


# ---------------------------------------------------------------------------
# Framework-owned env markers
# ---------------------------------------------------------------------------

#: Env marker naming the server's own ``mcp__<name>__`` tool-prefix identity.
#:
#: An MCP server process cannot work out which server it *is*: fastmcp reports
#: bare tool names (``phoebus_drive``), while every gate list, hook matcher and
#: permission string in the render is spelled fully-qualified
#: (``mcp__phoebus2__phoebus_drive``). In-process consumers — the audit
#: middleware first — qualify what they see by reading this marker.
#:
#: Distinct from two identities it resembles, and deliberately so:
#:
#: * the ``.mcp.json`` server KEY, which is what Claude Code launches the
#:   server under (equal in value here, but not the same contract: the key is
#:   the launcher's, this is the process's);
#: * ``OSPREY_SERVER_NAME``, the web-terminal panel id — which stays
#:   facility-pinnable, because pointing a clone at an existing panel tab is a
#:   legitimate deployment choice.
#:
#: **Unset is part of the contract, not a bug.** Two shapes of server never
#: receive it: a URL-based server gets no rendered env at all (``mcp.json.j2``
#: emits only ``{type, url}`` on its url branch — pinned by
#: ``tests/registry/test_mcp.py::test_render_mcp_json_url_server``),
#: and any server started outside a render (hand-run process, test harness,
#: a facility's own launcher) inherits whatever the shell had. In-process
#: consumers MUST therefore treat an unset marker as "leave tool names
#: unqualified" rather than assume presence: qualifying against a guessed
#: prefix would manufacture names that match no gate list, hook matcher or
#: permission string in the render, which is worse than an unqualified name.
TOOL_PREFIX_ENV = "OSPREY_MCP_TOOL_PREFIX"


# ── Audit-critical markers: framework-REMOVED, never spec-set ─────────────
#
# These say WHO acted, WHICH POSTURE the process is under, and WHERE that
# posture claim came from. Every one of them is assigned by a real assignment
# site outside the MCP spec path, and none of them has any legitimate reason
# to appear in a server spec's ``env:`` — a spec that could set one could make
# its own records claim another service's identity, present a writes posture
# inside a sandboxed session (lifting the middleware clamp AND filing a false
# ``posture=writes`` record), or claim a posture provenance it was never
# granted. So the spec path REMOVES them (post-merge, in
# :func:`_server_to_dict`), the exact mirror of the tool prefix's post-merge
# assignment.
#
# The set is spelled off its AUTHORITATIVE SOURCES, not off hand-picked
# markers: the whole identity ladder (``osprey.utils.identity``, every rung —
# stripping only the lower rung left the winning ``OSPREY_TERMINAL_USER``
# pinnable, which misrouted a server's entire ledger into an unmounted
# subdirectory) and the posture value itself (``osprey.audit.posture``). Both
# owners are stdlib-only leaves that import nothing from ``osprey`` beyond the
# envelope, so the registry can depend on them from anywhere without a cycle.
# The remaining three are spelled locally rather than imported: their owners
# live in ``osprey.interfaces.web_terminal`` and in the audit writer, and
# importing an interfaces module from the registry would risk an import cycle.
# The spellings are pinned against their owners by
# ``tests/registry/test_marker_nonpinnability.py``, so a rename there fails
# here instead of silently un-stripping a marker.

#: Which spawn path decided this session's posture — ``live``, ``spawn``,
#: ``process``, ``app``. Assigned at the three web-terminal spawn sites via
#: ``osprey.interfaces.web_terminal.operator_session.POSTURE_SOURCE_ENV``.
POSTURE_SOURCE_ENV = "OSPREY_POSTURE_SOURCE"

#: The session key a record joins on, exported into the session's children.
#: Assigned beside :data:`POSTURE_SOURCE_ENV` at the same spawn sites via
#: ``operator_session.POSTURE_SESSION_ENV``.
POSTURE_SESSION_ENV = "OSPREY_POSTURE_SESSION"

#: Names the maintenance writer for records made by the root maintenance
#: heredoc. Assigned ONLY as a per-command env prefix on that invocation, and
#: must stay absent everywhere else — an inherited value would misroute every
#: app-side record. Reserved here so the spec path already refuses it before
#: the writer that assigns it exists.
AUDIT_WRITER_ENV = "OSPREY_AUDIT_WRITER"

#: The audit-critical markers, in one tuple so a drift check can assert
#: membership rather than re-encode the list. This is the seam the gate-wiring
#: drift test imports. Spelled off the identity ladder and the posture
#: module's own name so a rung or a marker added THERE is stripped HERE without
#: anyone remembering to extend this tuple; ``test_marker_nonpinnability.py``
#: asserts both containments.
#:
#: ``OSPREY_EXECUTION_MODE`` is removed outright rather than narrowed: a
#: deployment-wide readonly posture legitimately arrives through the
#: container's ``environment:`` or a spawn site — never through a per-server
#: ``.mcp.json`` env block, which the process environment the session set
#: would otherwise be overridden by.
NON_PINNABLE_AUDIT_MARKERS: tuple[str, ...] = (
    *IDENTITY_ENV_LADDER,
    POSTURE_ENV_VAR,
    AUDIT_WRITER_ENV,
    POSTURE_SOURCE_ENV,
    POSTURE_SESSION_ENV,
)

#: Env markers a server spec's ``env:`` may not set — the framework owns them.
#:
#: Spec env otherwise WINS the env merge (both for extends clones and for
#: custom servers, whose spec env is copied verbatim), which is the documented
#: override contract and stays. These keys are the exception: they are settled
#: AFTER the merge in :func:`_server_to_dict`, so a spec value never reaches
#: the rendered ``.mcp.json``. :func:`_lint_framework_owned_env` warns rather
#: than silently discarding, so a facility learns its pin does nothing.
#:
#: Two classes, settled the same way and reported differently, because the
#: operator's next question differs: :data:`TOOL_PREFIX_ENV` is ASSIGNED after
#: the merge (the pin is overwritten with the framework's value), while every
#: entry of :data:`NON_PINNABLE_AUDIT_MARKERS` is REMOVED after it (the key is
#: gone from the rendered env entirely, and only a real assignment site
#: outside the spec path can put it back).
_FRAMEWORK_OWNED_SPEC_ENV: tuple[str, ...] = (TOOL_PREFIX_ENV, *NON_PINNABLE_AUDIT_MARKERS)

#: The subset of :data:`_FRAMEWORK_OWNED_SPEC_ENV` that is removed rather than
#: reassigned — the lint's message variant is keyed on this membership.
_REMOVED_SPEC_ENV: frozenset[str] = frozenset(NON_PINNABLE_AUDIT_MARKERS)


def _lint_framework_owned_env(name: str, spec: dict) -> None:
    """Warn when a server spec tries to pin a framework-owned env marker.

    Runs on EVERY spec — framework override, extends clone and custom server
    alike — because the post-merge settlement in :func:`_server_to_dict`
    covers all three and the operator deserves to hear about a pin that will
    be ignored on any of them.

    Args:
        name: The server name the spec is keyed under.
        spec: The raw config spec (already known to be a dict).
    """
    spec_env = spec.get("env")
    if not isinstance(spec_env, dict):
        # A malformed ``env:`` (list, string, null) is not this lint's problem;
        # it has nothing to pin.
        return
    for marker in _FRAMEWORK_OWNED_SPEC_ENV:
        if marker in spec_env:
            settlement = (
                "removed after the spec env merge"
                if marker in _REMOVED_SPEC_ENV
                else "assigned after the spec env merge"
            )
            logger.warning(
                "Server %r pins %s in its spec env — that marker is owned by the "
                "framework and is %s, so the pinned value %r is ignored; remove it "
                "from the spec",
                name,
                marker,
                settlement,
                spec_env[marker],
            )


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_servers(claude_code_config: dict, ctx: dict) -> list[dict]:
    """Resolve the full list of MCP servers from the registry + config overrides.

    Args:
        claude_code_config: The ``claude_code`` section of config.yml.
        ctx: Template context with derived values (project_root, confluence, etc.).

    Returns:
        List of plain dicts, each representing one server with keys:
        name, enabled, url, transport (``"http"``/``"sse"`` for URL servers,
        ``None`` for stdio), command, args, env, permissions_allow,
        permissions_ask, fixed_allow, hooks_pre, hooks_post, is_custom.
    """
    servers: dict[str, ServerDefinition] = {
        k: copy.deepcopy(v) for k, v in FRAMEWORK_SERVERS.items()
    }

    # ── Channel-finder pipeline tools ─────────────────────────
    # The channel-finder server's tool set is pipeline-specific. Render the
    # active pipeline's tools into permissions_allow so settings.json and the
    # agent frontmatter share one source of truth (no wildcard).
    cf_pipeline = ctx.get("channel_finder_pipeline")
    if cf_pipeline and "channel-finder" in servers:
        # A paradigm the registry does not know is a build error, not an empty
        # allow-list: silently allowing nothing would ship a channel-finder
        # server whose every tool is denied at run time.
        if cf_pipeline not in VALID_CHANNEL_FINDER_MODES:
            from osprey.services.channel_finder.core.exceptions import PipelineModeError

            raise PipelineModeError(
                f"Unknown channel_finder pipeline: {cf_pipeline!r}. "
                f"Valid modes are: {', '.join(VALID_CHANNEL_FINDER_MODES)}."
            )
        servers["channel-finder"].permissions_allow = list(
            CHANNEL_FINDER_TOOLS_BY_PIPELINE.get(cf_pipeline, [])
        )

    # ── Evaluate conditions ────────────────────────────────────
    for sdef in servers.values():
        if sdef.condition and not ctx.get(sdef.condition):
            sdef.default_enabled = False

    # ── New-format overrides: claude_code.servers ──────────────
    server_overrides = claude_code_config.get("servers", {})
    for name, spec in server_overrides.items():
        if not isinstance(spec, dict):
            continue
        _lint_framework_owned_env(name, spec)
        if name in servers:
            # Override existing framework server. Note: only 'enabled' applies
            # here — an 'extends' key on a framework name would silently shadow
            # a safety-configured definition, so it is rejected loudly instead.
            if "extends" in spec:
                logger.warning(
                    "Server %r is a framework server — its 'extends' key is ignored "
                    "(framework definitions cannot be shadowed); only 'enabled' applies",
                    name,
                )
            if spec.get("enabled") is False:
                servers[name].default_enabled = False
            elif spec.get("enabled") is True:
                servers[name].default_enabled = True
        elif "extends" in spec:
            # Second instance of a framework server (e.g. phoebus2 → phoebus).
            # Declared ⇒ enabled unless the spec says enabled: false (matches
            # the declared-custom-server convention below).
            if spec.get("enabled") is False:
                continue
            clone = build_extended_server(name, spec)
            if clone is not None:
                servers[name] = clone
        elif spec.get("enabled") is not False:
            # Custom server
            if not spec.get("command") and not spec.get("url"):
                # Never emit a broken {"command": ""} entry into .mcp.json —
                # e.g. a 'phoebus2: {enabled: true}' spec naming a server no
                # framework entry defines.
                logger.warning(
                    "Server %r has none of 'extends'/'command'/'url' — skipping "
                    "(a second framework-server instance is declared via "
                    "'extends: <framework-server>')",
                    name,
                )
                continue
            custom = _custom_server_from_spec(name, spec)
            if custom is not None:
                servers[name] = custom

    # ── Build output dicts ────────────────────────────────────
    result = []
    for sdef in servers.values():
        result.append(_server_to_dict(sdef, ctx))
    return result


# Extends-clone names are spliced into regex hook matchers, exact permission
# strings, and the startswith-matched prefixes of hook_config.json — restrict
# them to characters that are inert in all three contexts. '__' is additionally
# forbidden: it would corrupt osprey_approval's short-name extraction. A
# trailing '_' is also forbidden: 'controls_' yields the prefix
# 'mcp__controls___', which startswith-collides with 'mcp__controls__' and
# corrupts approval short-name extraction the same way.
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*(?<!_)$")


def build_extended_server(name: str, spec: dict) -> ServerDefinition | None:
    """Clone a framework ServerDefinition per an ``extends`` config spec.

    Supports second instances of framework servers without copy-pasting the
    registry entry::

        claude_code:
          servers:
            phoebus2:
              extends: phoebus
              env:
                PHOEBUS_BRIDGE_URL: "${PHOEBUS2_BRIDGE_URL:-http://127.0.0.1:7980}"

    Semantics:

    * Deep-copies the pristine ``FRAMEWORK_SERVERS[template]`` (never a mutated
      per-call copy — clones are independent of declaration order and of the
      template's own ``enabled`` override).
    * Enablement comes ONLY from the spec: declared ⇒ enabled unless
      ``enabled: false`` (the template's ``default_enabled`` must not leak
      through the copy — phoebus ships ``default_enabled=False``).
    * Rewrites every hook matcher that starts with the anchored prefix
      ``mcp__<template>__`` to ``mcp__<name>__`` (prefix splice only — a bare
      name replace would corrupt tool names like ``phoebus_drive``). Bare tool
      names in permission lists are left untouched.
    * Merges spec ``env`` over the template env (spec keys win; ``${...}``
      values pass through for runtime expansion, as everywhere else).
    * ``permissions.allow`` / ``permissions.ask`` replace the inherited lists
      when given, EXCEPT that the template's ``permissions_ask`` members can
      only be added to, never removed: the ask set doubles as the headless
      side-effect classifier (agent_runner.write_tools), so an override that
      promoted e.g. ``phoebus_drive`` into ``allow`` would silently ungate it
      under ``bypassPermissions``.

    Since ``is_external`` stays False, the clone renders exactly like the
    template (``python -m <module>``) and ``is_custom`` is False — it is listed
    as a regular (not "extra") server in the regen summary; accepted cosmetic
    gap.

    Returns:
        The cloned ServerDefinition, or ``None`` (after a logged warning) for
        invalid specs: bad clone name, unknown/non-framework extends target
        (chaining is not supported), a name shadowing a framework server, or a
        conditioned/dynamic template (e.g. channel-finder, whose module and
        permissions are resolved per-pipeline and cannot be cloned).
    """
    target = spec.get("extends")
    if not isinstance(name, str) or not _SERVER_NAME_RE.match(name) or "__" in name:
        logger.warning(
            "Invalid extends server name %r — must match [A-Za-z0-9][A-Za-z0-9_-]* "
            "without '__' and not ending in '_'; skipping",
            name,
        )
        return None
    if name in FRAMEWORK_SERVERS:
        logger.warning(
            "Extends server %r shadows a framework server of the same name — skipping",
            name,
        )
        return None
    # isinstance BEFORE the membership test: a non-string target (e.g.
    # ``extends: [phoebus]``) would TypeError on the unhashable dict lookup.
    if not isinstance(target, str) or target not in FRAMEWORK_SERVERS:
        logger.warning(
            "Unknown extends target %r for server %r — must name a framework server "
            "(chaining extends/custom servers is not supported); skipping",
            target,
            name,
        )
        return None
    template = FRAMEWORK_SERVERS[target]
    if template.condition:
        logger.warning(
            "Extends target %r is a conditioned/dynamic server — cloning is not "
            "supported; skipping server %r",
            target,
            name,
        )
        return None

    # Deep copy so the clone never shares HookRule/env objects with the template.
    clone = copy.deepcopy(template)
    clone.name = name
    clone.extends_of = target
    clone.default_enabled = spec.get("enabled") is not False

    # Anchored matcher rewrite: mcp__<template>__… → mcp__<name>__…
    old_prefix = f"mcp__{target}__"
    new_prefix = f"mcp__{name}__"
    for rule in (*clone.hooks_pre, *clone.hooks_post):
        if rule.matcher.startswith(old_prefix):
            rule.matcher = new_prefix + rule.matcher[len(old_prefix) :]

    # fixed_allow/fixed_ask hold fully-qualified mcp__<server>__ strings —
    # apply the same anchored prefix rewrite so they follow the clone.
    clone.fixed_allow = [
        new_prefix + entry[len(old_prefix) :] if entry.startswith(old_prefix) else entry
        for entry in clone.fixed_allow
    ]
    clone.fixed_ask = [
        new_prefix + entry[len(old_prefix) :] if entry.startswith(old_prefix) else entry
        for entry in clone.fixed_ask
    ]

    # Spec env merges over template env (spec keys win).
    merged_env = dict(clone.env)
    merged_env.update(spec.get("env") or {})
    # Instance identity follows the clone, like the matcher rewrite: unless the
    # spec pins OSPREY_SERVER_NAME explicitly, the clone advertises its OWN
    # name (inheriting the template's would make every instance signal the
    # template's web-terminal panel, e.g. phoebus2 focusing the phoebus tab).
    if "OSPREY_SERVER_NAME" not in (spec.get("env") or {}):
        merged_env["OSPREY_SERVER_NAME"] = name
    clone.env = merged_env

    # Optional permission overrides (add-only for the template's ask set).
    # Dedupe (order-preserving) so a duplicated entry cannot defeat the
    # single .remove() in the ask-union guard below.
    perms = spec.get("permissions") or {}
    if "allow" in perms:
        clone.permissions_allow = list(dict.fromkeys(perms.get("allow") or []))
    if "ask" in perms:
        clone.permissions_ask = list(dict.fromkeys(perms.get("ask") or []))
    for tool in template.permissions_ask:
        if tool not in clone.permissions_ask:
            logger.warning(
                "Server %r: override may not remove approval-gated tool %r inherited "
                "from %r — re-adding it to permissions.ask",
                name,
                tool,
                target,
            )
            clone.permissions_ask.append(tool)
        if tool in clone.permissions_allow:
            clone.permissions_allow.remove(tool)

    return clone


#: Valid ``transport`` values for URL-based custom servers.
VALID_TRANSPORTS = ("http", "sse")


def _custom_server_from_spec(name: str, spec: dict) -> ServerDefinition | None:
    """Build a ServerDefinition from a new-format config spec.

    Returns ``None`` (after a logged warning) for an invalid ``transport``
    value — silently defaulting a typo like ``trasnport: ssse`` to HTTP would
    ship a server that can never connect, so the spec is rejected loudly
    instead (matches the missing-command/url handling in resolve_servers).

    The name is validated like an ``extends`` clone name: tool names are
    ``mcp__<name>__<tool>``, so a ``__`` inside the name (or a trailing
    ``_``) would make every consumer that splits on the first ``__`` — the
    transcript reader, the display formatters — silently mis-attribute this
    server's calls to a differently-named server.
    """
    if not isinstance(name, str) or not _SERVER_NAME_RE.match(name) or "__" in name:
        logger.warning(
            "Invalid custom server name %r — must match [A-Za-z0-9][A-Za-z0-9_-]* "
            "without '__' and not ending in '_'; skipping",
            name,
        )
        return None

    perms = spec.get("permissions", {})

    transport = spec.get("transport", "http")
    if spec.get("url"):
        if transport not in VALID_TRANSPORTS:
            logger.warning(
                "Server %r has invalid transport %r — must be one of %s; skipping",
                name,
                transport,
                "/".join(VALID_TRANSPORTS),
            )
            return None
    elif "transport" in spec:
        # A command server is structurally stdio — there is no transport choice.
        logger.warning(
            "Server %r declares 'transport' but launches via 'command' — stdio "
            "servers have no transport choice; ignoring the key",
            name,
        )
        transport = "http"

    # Resolve pre-tool-use hook presets
    hooks_pre: list[HookRule] = []
    pre_presets = spec.get("hooks", {}).get("pre_tool_use", [])
    if pre_presets:
        resolved = []
        for preset in pre_presets:
            hook = HOOK_PRESETS.get(preset)
            if hook:
                resolved.append(hook)
            else:
                logger.warning("Unknown hook preset %r for server %r — skipping", preset, name)
        if resolved:
            hooks_pre = [HookRule(matcher=f"mcp__{name}__.*", hooks=resolved)]

    spec_env = spec.get("env")
    if spec_env is None:
        spec_env = {}
    elif not isinstance(spec_env, dict):
        # ``env: null`` or a list/scalar used to reach ``_server_to_dict``'s
        # ``.items()`` and crash the whole resolve. Fail closed on the SPEC
        # (no env) rather than on every server in the deployment.
        logger.warning(
            "Server %r has a malformed env: (expected a mapping, got %s) — ignoring it",
            name,
            type(spec_env).__name__,
        )
        spec_env = {}

    return ServerDefinition(
        name=name,
        module="",
        env=spec_env,
        is_external=True,
        external_command=spec.get("command", ""),
        external_args=spec.get("args", []),
        url=spec.get("url"),
        transport=transport,
        port=spec.get("port"),
        permissions_allow=perms.get("allow", []),
        permissions_ask=perms.get("ask", []),
        hooks_pre=hooks_pre,
        hooks_post=[_post_error(f"mcp__{name}__.*")]
        if perms.get("allow") or perms.get("ask")
        else [],
    )


def _server_to_dict(sdef: ServerDefinition, ctx: dict) -> dict:
    """Convert a ServerDefinition into a plain dict for templates."""
    # Resolve command / URL
    url = sdef.url
    if sdef.url:
        command = ""
        args = []
    elif sdef.is_external:
        command = sdef.external_command or ""
        args = [_resolve_placeholder(a, ctx) for a in sdef.external_args]
    else:
        command = ctx.get("current_python_env", "python")
        module = _resolve_placeholder(sdef.module, ctx)
        args = ["-m", module] + list(sdef.args_extra)

    # Resolve env placeholders
    env = {}
    for k, v in sdef.env.items():
        env[k] = _resolve_placeholder(v, ctx)

    # ── Framework-owned markers, settled POST-merge ───────────
    # This is the ONE site every launch path funnels through — framework
    # servers, extends clones, and custom specs alike — which is exactly why
    # the settlement lives here: spec env wins the merge everywhere upstream
    # (a clone merges it over the template, a custom server copies it
    # verbatim), so anything assigned before this point is pinnable from a
    # spec. Assigning after the resolution loop, unconditionally, is what
    # makes the tool-prefix identity non-pinnable — see TOOL_PREFIX_ENV and
    # the _lint_framework_owned_env warning that tells the operator so.
    # Removal comes first: the audit-critical markers have no framework value
    # to put here at all. Each one is assigned by a real site outside this path
    # (compose environment:, a spawn site, the maintenance heredoc), so a value
    # arriving through a spec could only ever be a spoof — a write-capable
    # clone claiming a read-only service's identity, or a session's posture
    # provenance it was never granted. Popping here rather than on the clone
    # path is what covers CUSTOM servers too, whose spec env is copied verbatim.
    for marker in NON_PINNABLE_AUDIT_MARKERS:
        env.pop(marker, None)

    # The value is the server's resolved name: the .mcp.json key it launches
    # under, hence the mcp__<name>__ prefix its tools carry.
    env[TOOL_PREFIX_ENV] = sdef.name

    # Convert hooks to plain dicts
    hooks_pre = [_hook_rule_to_dict(r) for r in sdef.hooks_pre]
    hooks_post = [_hook_rule_to_dict(r) for r in sdef.hooks_post]

    permissions_ask = list(sdef.permissions_ask)
    fixed_ask = list(sdef.fixed_ask)

    return {
        "name": sdef.name,
        "enabled": sdef.default_enabled,
        "url": url,
        # Transport only means something for URL servers; None for stdio so a
        # consumer can never mistake a command server for an HTTP one.
        "transport": sdef.transport if url else None,
        "command": command,
        "args": args,
        "env": env,
        "permissions_allow": list(sdef.permissions_allow),
        "permissions_ask": permissions_ask,
        "fixed_allow": list(sdef.fixed_allow),
        "fixed_ask": fixed_ask,
        "hooks_pre": hooks_pre,
        "hooks_post": hooks_post,
        "is_custom": sdef.is_external and sdef.name not in FRAMEWORK_SERVERS,
        "extends_of": sdef.extends_of,
    }


def _hook_rule_to_dict(rule: HookRule) -> dict:
    """Convert a HookRule to a plain dict."""
    return {
        "matcher": rule.matcher,
        "hooks": [{"type": h.type, "command": h.command, "timeout": h.timeout} for h in rule.hooks],
    }


def _resolve_placeholder(value: str, ctx: dict) -> str:
    """Resolve ``{key}`` placeholders in a string against ctx.

    Handles special cases:
    - ``{project_root}`` → ctx["project_root"]
    - ``{current_python_env}`` → ctx["current_python_env"]
    - ``{channel_finder_pipeline}`` → ctx["channel_finder_pipeline"]
    - ``${...}`` env-var references are left untouched (resolved at runtime)
    """
    if "{channel_finder_pipeline}" in value:
        value = value.replace(
            "{channel_finder_pipeline}",
            str(ctx.get("channel_finder_pipeline", "")),
        )

    if "{project_root}" in value:
        value = value.replace("{project_root}", str(ctx.get("project_root", "")))

    if "{current_python_env}" in value:
        value = value.replace(
            "{current_python_env}",
            str(ctx.get("current_python_env", "python")),
        )

    return value


# ---------------------------------------------------------------------------
# Agent resolution
# ---------------------------------------------------------------------------


def resolve_agents(
    claude_code_config: dict,
    ctx: dict,
    project_dir: Path | None = None,
    resolved_servers: list[dict] | None = None,
) -> list[dict]:
    """Resolve the full list of agents from the registry + config overrides.

    Args:
        claude_code_config: The ``claude_code`` section of config.yml.
        ctx: Template context.
        project_dir: Project root (for scanning custom agents).
        resolved_servers: Output of ``resolve_servers()`` (for server deps).

    Returns:
        List of plain dicts with keys: name, enabled, description, is_custom.
    """
    agents: dict[str, AgentDefinition] = {k: copy.deepcopy(v) for k, v in FRAMEWORK_AGENTS.items()}

    enabled_servers = set()
    if resolved_servers:
        enabled_servers = {s["name"] for s in resolved_servers if s["enabled"]}

    # ── Evaluate conditions ────────────────────────────────────
    for adef in agents.values():
        if adef.condition and not ctx.get(adef.condition):
            adef.default_enabled = False
        if adef.server_dependency and adef.server_dependency not in enabled_servers:
            adef.default_enabled = False

    # ── Config overrides: claude_code.agents ─────────────────
    agent_overrides = claude_code_config.get("agents", {})
    new_custom: list[AgentDefinition] = []
    for name, spec in agent_overrides.items():
        if not isinstance(spec, dict):
            continue
        if name in agents:
            if spec.get("enabled") is False:
                agents[name].default_enabled = False
            elif spec.get("enabled") is True:
                agents[name].default_enabled = True
        else:
            if spec.get("enabled") is not False:
                adef = _custom_agent_from_spec(name, spec)
                agents[name] = adef
                new_custom.append(adef)

    # ── Evaluate conditions (config-defined custom agents) ────
    for adef in new_custom:
        if adef.condition and not ctx.get(adef.condition):
            adef.default_enabled = False
        if adef.server_dependency and adef.server_dependency not in enabled_servers:
            adef.default_enabled = False

    # ── Auto-discover custom agents ───────────────────────────
    if project_dir:
        agents_dir = Path(project_dir) / ".claude" / "agents"
        if agents_dir.is_dir():
            for md_file in sorted(agents_dir.glob("*.md")):
                agent_name = md_file.stem
                if agent_name not in agents:
                    desc = _parse_agent_frontmatter(md_file)
                    agents[agent_name] = AgentDefinition(
                        name=agent_name,
                        description=desc,
                        is_custom=True,
                    )

    # ── Build output ──────────────────────────────────────────
    return [_agent_to_dict(a) for a in agents.values()]


def _agent_to_dict(adef: AgentDefinition) -> dict:
    return {
        "name": adef.name,
        "enabled": adef.default_enabled,
        "description": adef.description,
        "is_custom": adef.is_custom,
        "requires_ask_tools": sorted(adef.requires_ask_tools),
    }


def _custom_agent_from_spec(name: str, spec: dict) -> AgentDefinition:
    """Build an AgentDefinition from a config spec.

    Mirrors ``_custom_server_from_spec()`` for servers.
    """
    return AgentDefinition(
        name=name,
        template_path=spec.get("template_path"),
        condition=spec.get("condition"),
        server_dependency=spec.get("server_dependency"),
        default_enabled=spec.get("enabled", True),
        description=spec.get("description", ""),
        is_custom=True,
    )


def _parse_agent_frontmatter(path) -> str:
    """Extract description from YAML frontmatter of an agent .md file."""
    try:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            end = text.index("---", 3)
            frontmatter = text[3:end]
            for line in frontmatter.splitlines():
                if line.strip().startswith("description:"):
                    return line.split(":", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""
