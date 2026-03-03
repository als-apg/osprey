"""Unified server and agent registry for Claude Code integration.

Replaces scattered Jinja2 template guards with a data-driven registry.
Templates become generic loops; all server/agent metadata lives here.

Users extend Osprey through ``config.yml`` — no framework source changes needed.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Hook helpers (reduce repetition in FRAMEWORK_SERVERS)
# ---------------------------------------------------------------------------

_APPROVAL = HookEntry(
    command='python "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_approval.py"',
    timeout=5,
)
_WRITES_CHECK = HookEntry(
    command='python "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_writes_check.py"',
    timeout=5,
)
_LIMITS = HookEntry(
    command='python "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_limits.py"',
    timeout=10,
)
_ERROR_GUIDANCE = HookEntry(
    command='python "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_error_guidance.py"',
    timeout=5,
)
_CF_FEEDBACK = HookEntry(
    command='python "$CLAUDE_PROJECT_DIR/.claude/hooks/osprey_cf_feedback_capture.py"',
    timeout=10,
)


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
        permissions_allow=[],
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
    "python": ServerDefinition(
        name="python",
        module="osprey.mcp_server.python_executor",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
            "CONFIG_FILE": "{project_root}/config.yml",
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
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
    "workspace": ServerDefinition(
        name="workspace",
        module="osprey.mcp_server.workspace",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
        },
        permissions_allow=[
            "facility_description",
            "screenshot_capture",
            "list_windows",
            "manage_window",
            "submit_response",
            "data_list",
            "data_read",
            "artifact_get",
            "artifact_focus",
            "artifact_export",
            "graph_extract",
            "graph_compare",
            "graph_save_reference",
            "create_static_plot",
            "create_interactive_plot",
            "create_dashboard",
            "create_document",
            "artifact_save",
            "session_log",
            "session_summary",
            "archiver_downsample",
            "setup_inspect",
            "lattice_init",
            "lattice_state",
            "lattice_set_param",
            "lattice_refresh",
            "lattice_set_baseline",
        ],
        permissions_ask=["setup_patch"],
        hooks_post=[_post_error("mcp__workspace__.*")],
    ),
    "ariel": ServerDefinition(
        name="ariel",
        module="osprey.mcp_server.ariel",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
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
        hooks_post=[_post_error("mcp__ariel__.*")],
    ),
    "accelpapers": ServerDefinition(
        name="accelpapers",
        module="osprey.mcp_server.accelpapers",
        args_extra=["serve"],
        env={
            "ACCELPAPERS_TYPESENSE_HOST": "${ACCELPAPERS_TYPESENSE_HOST:-localhost}",
            "ACCELPAPERS_TYPESENSE_PORT": "${ACCELPAPERS_TYPESENSE_PORT:-8108}",
            "ACCELPAPERS_TYPESENSE_API_KEY": "${ACCELPAPERS_TYPESENSE_API_KEY:-accelpapers-dev}",
            "ACCELPAPERS_OLLAMA_URL": "${ACCELPAPERS_OLLAMA_URL:-http://localhost:11434}",
        },
        permissions_allow=[
            "papers_search",
            "papers_get",
            "papers_browse",
            "papers_list_conferences",
            "papers_search_author",
            "papers_stats",
        ],
        hooks_post=[_post_error("mcp__accelpapers__.*")],
    ),
    "matlab": ServerDefinition(
        name="matlab",
        module="osprey.mcp_server.matlab",
        args_extra=["serve"],
        env={
            "MATLAB_MML_DB": "${MATLAB_MML_DB:-}",
        },
        condition="matlab",
        permissions_allow=[
            "mml_search",
            "mml_get",
            "mml_browse",
            "mml_dependencies",
            "mml_path",
            "mml_list_groups",
            "mml_stats",
        ],
        hooks_post=[_post_error("mcp__matlab__.*")],
    ),
    "channel-finder": ServerDefinition(
        name="channel-finder",
        module="osprey.mcp_server.channel_finder_{channel_finder_pipeline}",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
        },
        condition="channel_finder_pipeline",
        fixed_allow=["mcp__channel-finder"],
        hooks_post=[
            HookRule(
                matcher="mcp__channel-finder__.*",
                hooks=[_ERROR_GUIDANCE, _CF_FEEDBACK],
            ),
        ],
    ),
    "direct-channel-finder": ServerDefinition(
        name="direct-channel-finder",
        module="osprey.mcp_server.direct_channel_finder",
        env={
            "OSPREY_CONFIG": "{project_root}/config.yml",
        },
        condition="direct_channel_finder",
        fixed_allow=["mcp__direct-channel-finder"],
        hooks_post=[_post_error("mcp__direct-channel-finder__.*")],
    ),
    "confluence": ServerDefinition(
        name="confluence",
        module="",  # external — not a Python module
        env={
            "CONFLUENCE_URL": "{confluence_url}",
            "CONFLUENCE_PERSONAL_TOKEN": "${CONFLUENCE_ACCESS_TOKEN:-}",
        },
        condition="confluence",
        fixed_allow=["mcp__confluence"],
        is_external=True,
        external_command="uvx",
        external_args=["--python=3.12", "mcp-atlassian"],
        hooks_post=[_post_error("mcp__confluence__.*")],
    ),
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
    "literature-search": AgentDefinition(
        name="literature-search",
        description="Searches the accelerator physics literature database.",
    ),
    "wiki-search": AgentDefinition(
        name="wiki-search",
        condition="confluence",
        description="Searches the facility wiki for documentation and procedures.",
    ),
    "matlab-search": AgentDefinition(
        name="matlab-search",
        condition="matlab",
        description="Searches the MATLAB Middle Layer codebase database.",
    ),
    "graph-analyst": AgentDefinition(
        name="graph-analyst",
        condition="deplot",
        description="Extracts data from graph images and compares against references.",
    ),
    "data-visualizer": AgentDefinition(
        name="data-visualizer",
        server_dependency="python",
        description=(
            "Creates plots, dashboards, and compiles LaTeX documents. "
            "You do NOT have visualization tools."
        ),
    ),
    "direct-channel-finder": AgentDefinition(
        name="direct-channel-finder",
        condition="direct_channel_finder",
        server_dependency="direct-channel-finder",
        description=(
            "Searches live PV databases. "
            "Use ONLY when explicitly asked to find PVs via direct query."
        ),
    ),
    "textbook-expert": AgentDefinition(
        name="textbook-expert",
        description="Looks up concepts, equations, and derivations in accelerator physics textbooks.",
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

    # ── Evaluate conditions ────────────────────────────────────
    for sdef in servers.values():
        if sdef.condition and not ctx.get(sdef.condition):
            sdef.default_enabled = False

    # ── New-format overrides: claude_code.servers ──────────────
    server_overrides = claude_code_config.get("servers", {})
    for name, spec in server_overrides.items():
        if name in servers:
            # Override existing framework server
            if isinstance(spec, dict) and spec.get("enabled") is False:
                servers[name].default_enabled = False
            elif isinstance(spec, dict) and spec.get("enabled") is True:
                servers[name].default_enabled = True
        else:
            # Custom server
            if isinstance(spec, dict) and spec.get("enabled") is not False:
                servers[name] = _custom_server_from_spec(name, spec)

    # ── Legacy format: disable_servers / extra_servers ─────────
    legacy_disable = claude_code_config.get("disable_servers", [])
    if legacy_disable:
        warnings.warn(
            "claude_code.disable_servers is deprecated. "
            "Use claude_code.servers: {name: {enabled: false}} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for name in legacy_disable:
            if name in servers:
                servers[name].default_enabled = False

    legacy_extra = claude_code_config.get("extra_servers", {})
    if legacy_extra:
        warnings.warn(
            "claude_code.extra_servers is deprecated. "
            "Use claude_code.servers: {name: {command, args, env}} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for name, spec in legacy_extra.items():
            if name not in servers:
                servers[name] = _custom_server_from_legacy(name, spec)

    # ── Build output dicts ────────────────────────────────────
    result = []
    for sdef in servers.values():
        result.append(_server_to_dict(sdef, ctx))
    return result


def _custom_server_from_spec(name: str, spec: dict) -> ServerDefinition:
    """Build a ServerDefinition from a new-format config spec."""
    perms = spec.get("permissions", {})
    return ServerDefinition(
        name=name,
        module="",
        env=spec.get("env", {}),
        is_external=True,
        external_command=spec.get("command", ""),
        external_args=spec.get("args", []),
        permissions_allow=perms.get("allow", []),
        permissions_ask=perms.get("ask", []),
        hooks_post=[_post_error(f"mcp__{name}__.*")]
        if perms.get("allow") or perms.get("ask")
        else [],
    )


def _custom_server_from_legacy(name: str, spec: dict) -> ServerDefinition:
    """Build a ServerDefinition from the old extra_servers format.

    Legacy extra_servers are raw JSON blobs passed through to mcp.json.
    They get a wildcard ask permission (matching old behavior).
    """
    return ServerDefinition(
        name=name,
        module="",
        env=spec.get("env", {}),
        is_external=True,
        external_command=spec.get("command", ""),
        external_args=spec.get("args", []),
        fixed_allow=[],
        permissions_ask=[],
        # Legacy extra_servers got a wildcard ask entry "mcp__name" in settings
        fixed_ask=[f"mcp__{name}"],
    )


def _server_to_dict(sdef: ServerDefinition, ctx: dict) -> dict:
    """Convert a ServerDefinition into a plain dict for templates."""
    # Resolve command
    if sdef.is_external:
        command = sdef.external_command or ""
        args = list(sdef.external_args)
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
    - ``{confluence_url}`` → ctx.get("confluence", {}).get("url", "")
    - ``${...}`` env-var references are left untouched (resolved at runtime)
    """
    if "{confluence_url}" in value:
        confluence_url = ""
        confluence = ctx.get("confluence")
        if isinstance(confluence, dict):
            confluence_url = confluence.get("url", "")
        value = value.replace("{confluence_url}", confluence_url)

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

    # ── New-format overrides: claude_code.agents ──────────────
    agent_overrides = claude_code_config.get("agents", {})
    for name, spec in agent_overrides.items():
        if name in agents:
            if isinstance(spec, dict) and spec.get("enabled") is False:
                agents[name].default_enabled = False
            elif isinstance(spec, dict) and spec.get("enabled") is True:
                agents[name].default_enabled = True

    # ── Legacy format: disable_agents ─────────────────────────
    legacy_disable = claude_code_config.get("disable_agents", [])
    if legacy_disable:
        warnings.warn(
            "claude_code.disable_agents is deprecated. "
            "Use claude_code.agents: {name: {enabled: false}} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for name in legacy_disable:
            if name in agents:
                agents[name].default_enabled = False

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
