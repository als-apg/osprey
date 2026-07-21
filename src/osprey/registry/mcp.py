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

logger = logging.getLogger(__name__)


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
    url: str | None = None  # HTTP/SSE transport URL (mutually exclusive with command)
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            "CONFIG_FILE": "{project_root}/config.yml",
            "EPICS_CA_ADDR_LIST": "${EPICS_CA_ADDR_LIST:-}",
        },
        permissions_allow=["channel_limits"],
        permissions_ask=["channel_write"],
        hooks_pre=[
            HookRule(
                matcher="mcp__controls__channel_write",
                hooks=[_WRITES_CHECK, _LIMITS, _APPROVAL],
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            "CONFIG_FILE": "{project_root}/config.yml",
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            "CONFIG_FILE": "{project_root}/config.yml",
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            # osprey.utils.config reads CONFIG_FILE (not OSPREY_CONFIG); set both
            # so the server resolves config even when launched with a CWD other
            # than the project dir (e.g. the dispatch worker's /app WORKDIR).
            "CONFIG_FILE": "{project_root}/config.yml",
        },
        permissions_allow=[
            "facility_description",
            "screenshot_capture",
            "list_windows",
            "manage_window",
            "submit_response",
            "data_list",
            "data_read",
            "data_delete",
            "artifact_get",
            "artifact_focus",
            "artifact_export",
            "create_static_plot",
            "create_interactive_plot",
            "create_dashboard",
            "create_document",
            "artifact_save",
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
            "switch_panel",
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": "{project_root}/config.yml",
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
        },
        permissions_allow=[
            "keyword_search",
            "semantic_search",
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
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
            "OSPREY_CONFIG": "{project_root}/config.yml",
            "CONFIG_FILE": "{project_root}/config.yml",
            "BLUESKY_BRIDGE_URL": "${BLUESKY_BRIDGE_URL:-http://127.0.0.1:8090}",
            "BLUESKY_LAUNCH_TOKEN": "${BLUESKY_LAUNCH_TOKEN:-}",
        },
        # Tool names resolve from osprey.bluesky_tool_names (the single source of
        # truth) so a rename there follows through every gate here by construction.
        permissions_allow=[
            bsky.GET_RUN,
            bsky.LIST_PLANS,
            bsky.LIST_RUNS,
            bsky.GET_RUN_DATA,
            # Draft tools (task 2.1) never touch hardware — editing the shared
            # plan draft only stages what a future launch_run/Launch plan click
            # might run, so like the read tools above they need no approval
            # prompt and carry no _WRITES_CHECK hook. clear_draft is
            # nonetheless auto-classified side-effecting by
            # agent_runner.write_tools (matches bsky.DESTRUCTIVE_MARKERS'
            # "clear") and blocked under the headless read-only floor
            # regardless of this allow-listing — acceptable, expected
            # posture; do not rename the tool to dodge it.
            bsky.GET_DRAFT,
            bsky.SET_DRAFT,
            bsky.CLEAR_DRAFT,
        ],
        # launch_run starts a real scan; stop_run is the safe direction
        # and must never be kill-switch-blocked, so it carries approval only.
        # write_plan/validate_plan (task 2.3) reach NO hardware
        # either way: write_plan only writes a file (never imports/execs
        # it), and validate_plan's dry run drives mock devices only, in a
        # subprocess with EPICS_CA_* neutralized — both work identically whether
        # control_system.writes_enabled is on or off, so like stop_run neither
        # carries _WRITES_CHECK. They get their own (distinct, independently
        # allowlistable) short-names rather than reusing launch_run/stop_run's
        # tier, since an operator may want to permit authoring/validating plan
        # bodies without also auto-approving launch_run/stop_run, or vice versa.
        permissions_ask=[
            bsky.LAUNCH_RUN,
            bsky.STOP_RUN,
            bsky.WRITE_PLAN,
            bsky.VALIDATE_PLAN,
        ],
        hooks_pre=[
            HookRule(
                matcher=bsky.matcher(bsky.LAUNCH_RUN),
                hooks=[_WRITES_CHECK, _APPROVAL],
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
    "channel-finder": ServerDefinition(
        name="channel-finder",
        module="osprey.mcp_server.channel_finder_{channel_finder_pipeline}",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
            # See osprey_workspace: osprey.utils.config reads CONFIG_FILE.
            "CONFIG_FILE": "{project_root}/config.yml",
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
        "query_channels",
        "statistics",
        "validate",
    ],
    "in_context": ["query_channels"],
}


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
    "pyat-specialist": AgentDefinition(
        name="pyat-specialist",
        server_dependency="python",
        description=(
            "Delegate to this agent when the user needs lattice/optics quantities "
            "computed from the accelerator model (orbit, tunes, beta functions, "
            "dispersion, response matrices) — it writes and executes pyAT code "
            "against the simulated ALS-U AR ring."
        ),
    ),
}


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
        name, enabled, command, args, env, permissions_allow, permissions_ask,
        fixed_allow, hooks_pre, hooks_post, is_custom.
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
                # e.g. a legacy 'phoebus2: {enabled: true}' spec left over from
                # when phoebus2 was a framework server.
                logger.warning(
                    "Server %r has none of 'extends'/'command'/'url' — skipping "
                    "(a second framework-server instance is declared via "
                    "'extends: <framework-server>')",
                    name,
                )
                continue
            servers[name] = _custom_server_from_spec(name, spec)

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


def _custom_server_from_spec(name: str, spec: dict) -> ServerDefinition:
    """Build a ServerDefinition from a new-format config spec."""
    perms = spec.get("permissions", {})

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

    return ServerDefinition(
        name=name,
        module="",
        env=spec.get("env", {}),
        is_external=True,
        external_command=spec.get("command", ""),
        external_args=spec.get("args", []),
        url=spec.get("url"),
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

    # Convert hooks to plain dicts
    hooks_pre = [_hook_rule_to_dict(r) for r in sdef.hooks_pre]
    hooks_post = [_hook_rule_to_dict(r) for r in sdef.hooks_post]

    permissions_ask = list(sdef.permissions_ask)
    fixed_ask = list(sdef.fixed_ask)

    return {
        "name": sdef.name,
        "enabled": sdef.default_enabled,
        "url": url,
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
