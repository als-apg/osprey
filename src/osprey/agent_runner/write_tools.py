"""Write-tool list loader for the OSPREY headless agent runner.

Provides a single entry point, ``load_write_tools``, that returns the list of
MCP tool names subject to the writes kill switch.  The list is read from the
project's generated ``hook_config.json``; on any failure the module falls back
to a replica of the canonical list in
``templates/claude_code/claude/hooks/osprey_hook_log.py`` so the runner
always fail-closes (i.e. never accidentally omits a gated tool).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from osprey.agent_runner.tool_names import FOREIGN_MCP_NAMESPACES, READ_ONLY_DENIED_BUILTINS

logger = logging.getLogger(__name__)

# Replica of FALLBACK_WRITE_TOOLS from osprey_hook_log.py, where the hooks'
# shared write-gate helpers keep it.
# DRIFT GUARD: the test suite (test_write_tools.py) asserts that this list
# equals the canonical one in the hook file — update both together.
_FALLBACK_WRITE_TOOLS = [
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
    "mcp__controls__channel_write",
    "mcp__phoebus__phoebus_drive",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
]


#: What :func:`load_approval_prefixes` answers when the render's hook config cannot
#: be read: every MCP tool, so nothing is offered a standing allow by mistake.
_FALLBACK_APPROVAL_PREFIXES: tuple[str, ...] = ("mcp__",)


def load_approval_prefixes(project_dir: Path) -> list[str]:
    """Return the tool-name prefixes the render's approval hook asks about.

    Reads ``approval_prefixes`` from ``<project_dir>/.claude/hooks/hook_config.json``,
    the list the approval hook itself reads to decide which calls it puts in front of
    a person. A call under one of these prefixes is asked about on every call, so a
    standing allow for it would either do nothing or let the hook's ask be skipped.
    Fails closed: a file that is absent, unreadable or malformed answers
    :data:`_FALLBACK_APPROVAL_PREFIXES`, which covers every MCP tool.
    """
    hook_config_path = project_dir / ".claude" / "hooks" / "hook_config.json"
    try:
        config = json.loads(hook_config_path.read_text(encoding="utf-8"))
        prefixes = config.get("approval_prefixes") if isinstance(config, dict) else None
        if isinstance(prefixes, list) and all(isinstance(p, str) and p for p in prefixes):
            return list(dict.fromkeys(prefixes))
        logger.warning("approval_prefixes malformed in %s — using fallback", hook_config_path)
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s (%s) — using fallback", hook_config_path, exc)
    return list(_FALLBACK_APPROVAL_PREFIXES)


def load_write_tools(project_dir: Path) -> list[str]:
    """Return the MCP tool names that are subject to the writes kill switch.

    Reads ``<project_dir>/.claude/hooks/hook_config.json`` and returns the
    ``write_tools`` list.  Falls back to ``_FALLBACK_WRITE_TOOLS`` when the
    file is absent, unreadable, contains malformed JSON, or the ``write_tools``
    key is missing or empty.

    The returned list always contains the full canonical write-tool floor
    (``_FALLBACK_WRITE_TOOLS`` — every write-gated tool the framework ships):
    the fallback includes them by construction, and
    any member a loaded config omits is appended (with a warning) rather than
    trusted to be absent. This matters because ``disallowed_tools`` is the sole
    write guard under ``permission_mode=bypassPermissions`` (the approval hook
    is inert there), so a gated tool dropped from the project's ``write_tools``
    must not silently become callable. Disallowing a tool a project does not
    register is a harmless no-op, so the floor has no downside.

    Args:
        project_dir: Root of the OSPREY project (the directory that contains
            ``.claude/``).

    Returns:
        List of MCP tool-name strings that gate the writes kill switch, always
        a superset of ``_FALLBACK_WRITE_TOOLS``.
    """
    hook_config_path = project_dir / ".claude" / "hooks" / "hook_config.json"
    try:
        raw = hook_config_path.read_text(encoding="utf-8")
        config = json.loads(raw)
        tools = config.get("write_tools") or []
        if tools and not isinstance(tools, list):
            # A scalar (e.g. a bare string) would be character-split by list(),
            # silently dropping every gated tool. Fail closed to the fallback.
            logger.warning(
                "write_tools in %s is %s, not a list — using fallback",
                hook_config_path,
                type(tools).__name__,
            )
            tools = []
        if tools:
            result = list(tools)
            # Hard-floor every canonical write tool: a loaded config that omits
            # one would otherwise leave it callable under bypassPermissions.
            for required in _FALLBACK_WRITE_TOOLS:
                if required not in result:
                    logger.warning(
                        "%s is absent from write_tools in %s — injecting it "
                        "(write-safety floor; config drift)",
                        required,
                        hook_config_path,
                    )
                    result.append(required)
            return result
        logger.warning(
            "write_tools key missing or empty in %s — using fallback",
            hook_config_path,
        )
    except FileNotFoundError:
        logger.debug("hook_config.json not found at %s — using fallback", hook_config_path)
    except json.JSONDecodeError as exc:
        logger.warning(
            "malformed JSON in %s (%s) — using fallback",
            hook_config_path,
            exc,
        )
    except OSError as exc:
        logger.warning(
            "could not read %s (%s) — using fallback",
            hook_config_path,
            exc,
        )

    return list(_FALLBACK_WRITE_TOOLS)


def _registry_side_effect_tools() -> list[str]:
    """Side-effecting MCP tool names declared in the framework registry.

    Applies the registry's classifier,
    :func:`osprey.registry.mcp.server_side_effect_tools`, to every framework
    server. The registry owns the classification; this floor and the plan-mode
    deny list are its two consumers.
    """
    from osprey.registry.mcp import FRAMEWORK_SERVERS, server_side_effect_tools

    out: list[str] = []
    for server in FRAMEWORK_SERVERS.values():
        for name in server_side_effect_tools(server):
            if name not in out:
                out.append(name)
    return out


def _custom_server_side_effect_tools(project_dir: Path) -> list[str]:
    """Side-effecting tools from facility-defined custom MCP servers.

    Framework servers are covered by :func:`_registry_side_effect_tools`, but a
    project's ``config.yml`` ``claude_code.servers`` block can declare custom
    servers whose write tools never reach ``hook_config.json`` ``write_tools``
    (only writes-check-gated matchers do) and are invisible to the static
    registry walk. The documented pattern marks them with ``permissions.ask``::

        claude_code:
          servers:
            my-server:
              hooks: {pre_tool_use: [approval]}
              permissions: {allow: [read_data], ask: [write_data]}

    So for each enabled custom server this emits ``mcp__<server>__<tool>`` for
    every ``permissions.ask`` tool (exact, preserving the server's reads). If a
    custom server writes-check-gates *all* its tools (``pre_tool_use:
    [writes_check]``), it is blocked server-level (``mcp__<server>``) — the
    facility marked the whole server write-gated, and the per-tool regex matcher
    the registry would emit (``mcp__<server>__.*``) is a no-op in the SDK's
    exact-name/server-level disallow engine.

    An ``extends`` spec (second instance of a framework server) is classified
    from the RESOLVED clone via the registry's
    :func:`~osprey.registry.mcp.server_side_effect_tools`, so it
    inherits the template's ask/destructive-allow/writes-check surface with the
    ``mcp__<template>__`` → ``mcp__<name>__`` matcher rewrite. If the extends
    target cannot be resolved (typo, version skew, config edited after build)
    this fails CLOSED with a server-level ``mcp__<name>`` disallow — a stale
    ``.mcp.json`` from a previous build can still launch the server
    (``enableAllProjectMcpServers: true``), so an unresolvable spec must block
    the whole server rather than contribute nothing. A spec with NONE of
    extends/command/url (e.g. ``phoebus2: {enabled: true}``) fails closed the
    same way.

    Returns an empty list when ``config.yml`` is absent or unreadable.
    """
    try:
        cfg = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return []

    from osprey.registry.mcp import (
        FRAMEWORK_SERVERS,
        build_extended_server,
        is_destructive_tool,
        server_side_effect_tools,
    )

    servers = (cfg.get("claude_code") or {}).get("servers") or {}
    if not isinstance(servers, dict):
        return []

    out: list[str] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict) or spec.get("enabled") is False:
            continue
        if name in FRAMEWORK_SERVERS:
            continue  # framework server (or override) — covered by the registry walk
        if "extends" in spec:
            clone = build_extended_server(name, spec)
            if clone is None:
                out.append(f"mcp__{name}")  # fail closed (see docstring)
            else:
                out.extend(server_side_effect_tools(clone))
            continue
        if not spec.get("command") and not spec.get("url"):
            # NONE of extends/command/url (e.g. 'phoebus2: {enabled: true}'):
            # resolve_servers skips such a spec, so it never renders
            # into .mcp.json — but a stale .mcp.json from a previous build can
            # still launch a server by this name. Fail closed exactly like the
            # unresolvable-extends path above: block the whole server.
            out.append(f"mcp__{name}")
        perms = spec.get("permissions") or {}
        if not isinstance(perms, dict):
            perms = {}
        if "writes" in spec:
            out.extend(_declared_write_tools(name, spec, perms))
        out.extend(f"mcp__{name}__{tool}" for tool in perms.get("ask") or [])
        # Destructive-named tools a custom server auto-allows (mirrors the
        # framework destructive-allow handling) — block them even though the
        # facility put them in ``allow``.
        out.extend(
            f"mcp__{name}__{tool}" for tool in perms.get("allow") or [] if is_destructive_tool(tool)
        )
        pre = (spec.get("hooks") or {}).get("pre_tool_use") or []
        if "writes_check" in pre:
            # Whole server is write-gated; block it server-level (the only form
            # the SDK disallow engine honors for "all tools of a server").
            out.append(f"mcp__{name}")
    return out


def _declared_write_tools(name: str, spec: dict, perms: dict) -> list[str]:
    """Exact names a custom spec's ``writes:`` list gates (C-FW).

    Uses the registry's own classifier, so the headless block names exactly the
    tools the render's per-tool writes-check rules gate: the declared tools plus
    the warning window (undeclared asked or destructive-named tools). A
    declaring server's other tools stay callable. A list that cannot be read,
    or one combined with the whole-server ``writes_check`` preset (a spec the
    registry refuses), fails closed with a server-level ``mcp__<name>``.
    """
    from osprey.registry.mcp import _declared_write_rule_tools, _spec_writes

    pre = (spec.get("hooks") or {}).get("pre_tool_use") or []
    try:
        writes = _spec_writes(spec)
        if writes is None:
            return []
        if "writes_check" in pre:
            return [f"mcp__{name}"]
        return [f"mcp__{name}__{tool}" for tool in _declared_write_rule_tools(name, writes, perms)]
    except (TypeError, ValueError, AttributeError):
        return [f"mcp__{name}"]


#: Suffix of a whole-server hook matcher. The SDK's disallow engine honors only
#: exact names and the server-level ``mcp__<srv>`` form, so the regex is a no-op
#: there and is translated before it reaches the list.
_WHOLE_SERVER_SUFFIX = "__.*"


def _disallow_form(tool: str) -> str:
    """``mcp__<srv>__.*`` becomes ``mcp__<srv>``; any other name is unchanged."""
    if tool.startswith("mcp__") and tool.endswith(_WHOLE_SERVER_SUFFIX):
        return tool[: -len(_WHOLE_SERVER_SUFFIX)]
    return tool


def _string_list(value: object) -> list[str]:
    """``value`` when it is a list, keeping only its non-empty strings; else ``[]``.

    A scalar would otherwise be character-split by iteration.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _rendered_facility_write_tools(project_dir: Path) -> list[str]:
    """The render's facility write set plus its appended ``control_system.write_tools``.

    Read from ``hook_config.json``: ``facility_write_tools`` (rendered verbatim
    from :func:`osprey.registry.mcp.facility_write_tools`, the only producer of
    the facility set) and ``control_system_write_tools``. Whole-server matchers
    are translated to the server-level form. Answers ``[]`` when the file is
    absent or unreadable; the ``config.yml`` path and the write-tool floor still
    apply then.
    """
    hook_config_path = project_dir / ".claude" / "hooks" / "hook_config.json"
    try:
        config = json.loads(hook_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(config, dict):
        return []
    out: list[str] = []
    for key in ("facility_write_tools", "control_system_write_tools"):
        raw = config.get(key)
        if raw is not None and not isinstance(raw, list):
            logger.warning("%s in %s is not a list — ignored", key, hook_config_path)
        out.extend(_disallow_form(tool) for tool in _string_list(raw))
    return out


def _config_control_system_write_tools(project_dir: Path) -> list[str]:
    """``control_system.write_tools`` from ``config.yml``, in disallow form.

    Read directly so the entries stay blocked when the rendered hook config is
    missing or stale. Answers ``[]`` when ``config.yml`` is absent or unreadable.
    """
    try:
        cfg = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return []
    control_system = cfg.get("control_system") if isinstance(cfg, dict) else None
    if not isinstance(control_system, dict):
        return []
    return [_disallow_form(tool) for tool in _string_list(control_system.get("write_tools"))]


def read_only_disallowed_tools(project_dir: Path) -> list[str]:
    """Return the full ``disallowed_tools`` set for a headless read-only agent run.

    This is the set ``osprey query`` passes to the SDK as ``disallowed_tools``.
    Under ``permission_mode=bypassPermissions`` that list is the *sole* effective
    write guard (the allow/deny/ask permission layer is inert), so it must cover
    every write/exec/egress tool — built-in *and* MCP — or the read-only
    guarantee is hollow.

    It is the union of:

    * :data:`~osprey.agent_runner.tool_names.READ_ONLY_DENIED_BUILTINS` —
      built-in ``Bash`` (arbitrary shell, incl. hardware writes via ``caput``),
      ``Write``, ``Edit``, etc.
    * :data:`~osprey.agent_runner.tool_names.FOREIGN_MCP_NAMESPACES` — every
      Claude Code plugin's and claude.ai connector's MCP tools.
    * :func:`load_write_tools` — the project's hardware-write kill-switch tools
      (facility-custom writes included, fail-closed fallback).
    * :func:`_registry_side_effect_tools` — the registry classifier applied to
      every framework server: approval-required actions (``permissions_ask`` +
      ``fixed_ask``), destructive auto-allowed tools, and writes-check-gated
      tools (so the side-effect surface tracks the registry as the single
      source of truth, not a hand-maintained copy).
    * :func:`_custom_server_side_effect_tools` — write tools of facility-defined
      custom MCP servers declared in the project's ``config.yml``: each declared
      ``writes:`` tool exactly, ``permissions.ask`` tools, destructive-marker
      names, and whole-server writes-check servers as ``mcp__<srv>``. A
      declaring server's other tools stay callable.
    * :func:`_rendered_facility_write_tools` — the render's
      ``facility_write_tools`` and appended ``control_system_write_tools``.
    * :func:`_config_control_system_write_tools` — ``control_system.write_tools``.

    Args:
        project_dir: Root of the OSPREY project (the directory containing
            ``.claude/``).

    Returns:
        Deduplicated list of every tool blocked in the read-only path.
    """
    result = load_write_tools(project_dir)
    for tool in (
        *READ_ONLY_DENIED_BUILTINS,
        *FOREIGN_MCP_NAMESPACES,
        *_registry_side_effect_tools(),
        *_custom_server_side_effect_tools(project_dir),
        *_rendered_facility_write_tools(project_dir),
        *_config_control_system_write_tools(project_dir),
    ):
        if tool not in result:
            result.append(tool)
    return result
