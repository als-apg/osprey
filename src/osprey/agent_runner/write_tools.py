"""Write-tool list loader for the OSPREY headless agent runner.

Provides a single entry point, ``load_write_tools``, that returns the list of
MCP tool names subject to the writes kill switch.  The list is read from the
project's generated ``hook_config.json``; on any failure the module falls back
to a replica of the canonical list in
``templates/claude_code/claude/hooks/osprey_writes_check.py`` so the runner
always fail-closes (i.e. never accidentally omits a gated tool).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from osprey.bluesky_tool_names import DESTRUCTIVE_MARKERS

logger = logging.getLogger(__name__)

# Replica of _FALLBACK_WRITE_TOOLS from osprey_writes_check.py.
# DRIFT GUARD: the test suite (test_write_tools.py) asserts that this list
# equals the canonical one in the hook file — update both together.
_FALLBACK_WRITE_TOOLS = [
    "mcp__controls__channel_write",
    "mcp__python__execute",
]

# Built-in (non-MCP) Claude Code tools that can write to disk, execute shell
# commands, or reach the network. Under ``permission_mode=bypassPermissions``
# the settings.json allow/deny/ask layer is INERT, so for the headless,
# read-only ``osprey query`` path these MUST be blocked via ``disallowed_tools``
# or the read-only guarantee is hollow — e.g. ``Bash`` could ``caput`` a PV
# (a hardware write) or ``rm`` files, entirely bypassing the MCP write guard.
# This is a superset of the built-in (non-``mcp__``) entries in
# settings.json.j2's ``deny_defaults``; test_write_tools.py guards that the
# headless floor never drifts below the interactive deny policy.
_BUILTIN_UNSAFE_TOOLS = [
    "Bash",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
]


def load_write_tools(project_dir: Path) -> list[str]:
    """Return the MCP tool names that are subject to the writes kill switch.

    Reads ``<project_dir>/.claude/hooks/hook_config.json`` and returns the
    ``write_tools`` list.  Falls back to ``_FALLBACK_WRITE_TOOLS`` when the
    file is absent, unreadable, contains malformed JSON, or the ``write_tools``
    key is missing or empty.

    The returned list always contains the full canonical write-tool floor
    (``_FALLBACK_WRITE_TOOLS`` — ``mcp__controls__channel_write`` *and*
    ``mcp__python__execute``): the fallback includes them by construction, and
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


# Substrings in a tool name that mark it as destroying stored state. Used to
# auto-detect destructive tools that sit in an auto-approve (``allow``) list —
# e.g. the workspace server's data_delete/artifact_delete/artifact_delete_all,
# which are safe-to-auto-approve *interactively* (a human drives) but must not
# run in a headless read-only query. Sourced from the shared Bluesky tool-name
# module so the marker vocabulary has a single home: a rename there (e.g.
# clear_plan_draft → clear_draft) cannot silently detach a tool from this floor.
_DESTRUCTIVE_MARKERS = DESTRUCTIVE_MARKERS

# Side-effecting MCP tools the registry walk below cannot see because they are
# in NO registry permission list at all: the python server registers both
# ``execute`` (gated) and ``execute_file`` (unlisted), and SDK disallow matching
# is by exact tool name — so disallowing ``mcp__python__execute`` does NOT cover
# ``execute_file``, which runs arbitrary Python. (Destructive auto-allowed tools
# such as the workspace deletes are handled generically by the destructive-name
# scan in ``_registry_side_effect_tools``, not hard-coded here.) Keep this in
# sync with server tool modules; test_write_tools.py pins it.
_EXTRA_SIDE_EFFECT_TOOLS = [
    "mcp__python__execute_file",
]


def _is_destructive(tool: str) -> bool:
    """True if a tool's name implies it removes/destroys stored state."""
    lowered = tool.lower()
    return any(marker in lowered for marker in _DESTRUCTIVE_MARKERS)


def _server_side_effect_tools(server) -> list[str]:
    """Side-effecting MCP tool names declared by ONE ServerDefinition.

    Shared classifier for the framework-registry walk and the config-declared
    ``extends`` clone path — one implementation, so the two can never drift.
    Unions:

    * ``permissions_ask`` + ``fixed_ask`` — the curated "needs approval because
      it acts" sets (e.g. ``entry_create``, ``draft_concept``, ``setup_patch``,
      ``channel_write``, ``execute``). Both render into the interactive ``ask``
      list, so both are side-effecting.
    * destructive-named tools in ``permissions_allow`` + ``fixed_allow`` — tools
      auto-approved interactively but whose name implies destruction (e.g.
      ``data_delete``). See ``_is_destructive``.
    * every ``hooks_pre`` matcher gated by the writes-check hook (hardware
      writes).

    Deliberately NOT keyed on ``_APPROVAL``: that hook also wraps audited
    *reads* (``channel_read``, ``archiver_read``), which a read-only query must
    still be able to call.

    Does NOT include ``_EXTRA_SIDE_EFFECT_TOOLS`` (tools in no permission list
    at all) — callers union those separately: the framework walk via
    :func:`read_only_disallowed_tools`, extends clones via
    :func:`_extra_side_effect_tools_for`.

    Returns names as ``mcp__<server>__<tool>`` (writes-check matchers verbatim).
    """
    from osprey.registry.mcp import _WRITES_CHECK

    out: list[str] = []
    for tool in (*server.permissions_ask, *server.fixed_ask):
        out.append(f"mcp__{server.name}__{tool}")
    for tool in (*server.permissions_allow, *server.fixed_allow):
        if _is_destructive(tool):
            out.append(f"mcp__{server.name}__{tool}")
    for rule in server.hooks_pre:
        if _WRITES_CHECK in rule.hooks:
            out.append(rule.matcher)
    return out


def _extra_side_effect_tools_for(instance_name: str, template_name: str) -> list[str]:
    """``_EXTRA_SIDE_EFFECT_TOOLS`` entries owned by one framework template,
    rewritten to an instance's prefix.

    The extras list is keyed by framework-server name (``mcp__python__…``), but
    SDK disallow matching is by EXACT tool name — an ``extends`` clone launches
    the SAME tool module under a new prefix, so the template's extras must
    follow the clone as ``mcp__<instance>__<tool>`` or the clone would keep an
    unlisted exec tool (e.g. ``execute_file``) callable in read-only mode.
    """
    old = f"mcp__{template_name}__"
    new = f"mcp__{instance_name}__"
    return [new + tool[len(old) :] for tool in _EXTRA_SIDE_EFFECT_TOOLS if tool.startswith(old)]


def _registry_side_effect_tools() -> list[str]:
    """Side-effecting MCP tool names declared in the framework registry.

    Applies :func:`_server_side_effect_tools` to every framework server.
    """
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    out: list[str] = []
    for server in FRAMEWORK_SERVERS.values():
        for name in _server_side_effect_tools(server):
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
    from the RESOLVED clone via :func:`_server_side_effect_tools`, so it
    inherits the template's ask/destructive-allow/writes-check surface with the
    ``mcp__<template>__`` → ``mcp__<name>__`` matcher rewrite, plus the
    template's ``_EXTRA_SIDE_EFFECT_TOOLS`` entries rewritten the same way
    (:func:`_extra_side_effect_tools_for`). If the extends
    target cannot be resolved (typo, version skew, config edited after build)
    this fails CLOSED with a server-level ``mcp__<name>`` disallow — a stale
    ``.mcp.json`` from a previous build can still launch the server
    (``enableAllProjectMcpServers: true``), so an unresolvable spec must block
    the whole server rather than contribute nothing. A spec with NONE of
    extends/command/url (legacy form ``phoebus2: {enabled: true}``) fails
    closed the same way.

    Returns an empty list when ``config.yml`` is absent or unreadable.
    """
    try:
        cfg = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return []

    from osprey.registry.mcp import FRAMEWORK_SERVERS, build_extended_server

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
                out.extend(_server_side_effect_tools(clone))
                # The template's hard-coded extras (side-effect tools in NO
                # registry list, e.g. python's execute_file) must follow the
                # clone with the rewritten prefix — the registry classifier
                # cannot see them, and exact-name disallow matching means the
                # template's own entry does not cover the clone.
                out.extend(_extra_side_effect_tools_for(name, spec["extends"]))
            continue
        if not spec.get("command") and not spec.get("url"):
            # NONE of extends/command/url (legacy form 'phoebus2: {enabled:
            # true}'): resolve_servers skips such a spec, so it never renders
            # into .mcp.json — but a stale .mcp.json from a previous build can
            # still launch a server by this name. Fail closed exactly like the
            # unresolvable-extends path above: block the whole server.
            out.append(f"mcp__{name}")
        perms = spec.get("permissions") or {}
        for tool in perms.get("ask") or []:
            out.append(f"mcp__{name}__{tool}")
        # Destructive-named tools a custom server auto-allows (mirrors the
        # framework destructive-allow handling) — block them even though the
        # facility put them in ``allow``.
        for tool in perms.get("allow") or []:
            if _is_destructive(tool):
                out.append(f"mcp__{name}__{tool}")
        pre = (spec.get("hooks") or {}).get("pre_tool_use") or []
        if "writes_check" in pre:
            # Whole server is write-gated; block it server-level (the only form
            # the SDK disallow engine honors for "all tools of a server").
            out.append(f"mcp__{name}")
    return out


def read_only_disallowed_tools(project_dir: Path) -> list[str]:
    """Return the full ``disallowed_tools`` set for a headless read-only agent run.

    This is the set ``osprey query`` passes to the SDK as ``disallowed_tools``.
    Under ``permission_mode=bypassPermissions`` that list is the *sole* effective
    write guard (the allow/deny/ask permission layer is inert), so it must cover
    every write/exec/egress tool — built-in *and* MCP — or the read-only
    guarantee is hollow.

    It is the union of:

    * ``_BUILTIN_UNSAFE_TOOLS`` — built-in ``Bash`` (arbitrary shell, incl.
      hardware writes via ``caput``), ``Write``, ``Edit``, etc.
    * :func:`load_write_tools` — the project's hardware-write kill-switch tools
      (facility-custom writes included, fail-closed fallback).
    * :func:`_registry_side_effect_tools` — every framework server's
      approval-required actions (``permissions_ask`` + ``fixed_ask``),
      destructive auto-allowed tools, and writes-check-gated tools (so the
      side-effect surface tracks the registry as the single source of truth, not
      a hand-maintained copy).
    * ``_EXTRA_SIDE_EFFECT_TOOLS`` — side-effecting tools in NO registry
      permission list (currently ``mcp__python__execute_file``); ``extends``
      clones of the owning template get these rewritten to their own prefix
      via the custom-server path below.
    * :func:`_custom_server_side_effect_tools` — write tools of facility-defined
      custom MCP servers declared in the project's ``config.yml``.

    Args:
        project_dir: Root of the OSPREY project (the directory containing
            ``.claude/``).

    Returns:
        Deduplicated list of every tool blocked in the read-only path.
    """
    result = load_write_tools(project_dir)
    for tool in (
        *_BUILTIN_UNSAFE_TOOLS,
        *_registry_side_effect_tools(),
        *_EXTRA_SIDE_EFFECT_TOOLS,
        *_custom_server_side_effect_tools(project_dir),
    ):
        if tool not in result:
            result.append(tool)
    return result
