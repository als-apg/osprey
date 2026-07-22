"""Derive the auto ``mcp_servers`` health category from the facility config.

Synthesizes a :class:`~osprey.health.config.CategoryRecord` of
:class:`~osprey.health.config.CheckSpec`\\ s — one ``mcp`` probe per configured
MCP server — from the ``claude_code.servers`` block that ``osprey build`` writes
into ``config.yml``. This lets the framework check every wired MCP endpoint's
reachability (and, where declared, its tool surface) without the operator
hand-authoring a ``health.categories.mcp_servers`` block.

The single entry point is :func:`derive_mcp_servers`. It is a pure function of
the parsed :class:`~osprey.health.config.HealthSettings` and the
``${VAR}``-resolved config mapping; wiring the derived record into the merged
record set is the caller's concern (``build_records``), not this module's.

Parsing is deliberately defensive: any malformed server entry is skipped
silently rather than raising, because the config this consumes has already been
validated for the concerns the health parser owns — a shape surprise here should
degrade the derived category, never fail the whole health run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .config import (
    DEFAULT_PROBE_TIMEOUTS,
    CategoryRecord,
    CheckSpec,
    Cost,
    HealthSettings,
    resolve_callable_timeout_s,
)
from .models import Status

#: Category name of the auto-derived MCP-server health category.
_CATEGORY_NAME = "mcp_servers"


def _in_container() -> bool:
    """Best-effort detection of whether this process runs inside a container.

    Mirrors ``osprey.cli.claude_code_telemetry._running_in_container`` (a copy,
    not an import, to keep the health package free of a CLI dependency): a
    Docker-injected ``/.dockerenv`` marker file, or an explicit
    ``OSPREY_IN_CONTAINER`` environment flag set by the deploy environment.
    """
    return os.path.exists("/.dockerenv") or bool(os.environ.get("OSPREY_IN_CONTAINER"))


def _resolve_url_key(settings: HealthSettings) -> str:
    """Pick which ``network.*`` URL the derived probes target.

    An explicit ``health.auto.mcp.url_key`` always wins (even a ``"host_url"``
    choice while containerized — an explicit operator override is honored as
    written); otherwise the container-aware default applies: ``"docker_url"``
    inside a container, ``"host_url"`` on a host.
    """
    if settings.auto.url_key_explicit:
        return settings.auto.url_key
    return "docker_url" if _in_container() else "host_url"


def _expect_tools(entry: Mapping[str, Any]) -> list[str]:
    """Return the ordered, de-duplicated union of ``permissions.{allow,ask}``.

    Non-mapping ``permissions`` or non-list ``allow``/``ask`` values are treated
    as absent, and non-string / empty items are dropped. An empty union means the
    probe should perform a reachability/handshake check only (no tool assertion).
    """
    permissions = entry.get("permissions")
    if not isinstance(permissions, Mapping):
        return []

    tools: list[str] = []
    seen: set[str] = set()
    for key in ("allow", "ask"):
        values = permissions.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, str) and item and item not in seen:
                seen.add(item)
                tools.append(item)
    return tools


def _check_for_entry(name: str, entry: Any, url_key: str) -> CheckSpec | None:
    """Build the ``mcp`` :class:`CheckSpec` for one server entry, or ``None``.

    A qualifying entry is a mapping carrying a non-empty top-level ``url`` and a
    ``network`` mapping whose ``network[url_key]`` is a non-empty string. Any
    other shape returns ``None`` (skipped silently by the caller).
    """
    if not isinstance(entry, Mapping):
        return None
    if not entry.get("url"):
        return None

    network = entry.get("network")
    if not isinstance(network, Mapping):
        return None

    probe_url = network.get(url_key)
    if not isinstance(probe_url, str) or not probe_url:
        return None

    params: dict[str, Any] = {"url": probe_url}
    expect_tools = _expect_tools(entry)
    if expect_tools:
        params["expect_tools"] = expect_tools

    return CheckSpec(
        name=name,
        type="mcp",
        params=params,
        timeout_s=DEFAULT_PROBE_TIMEOUTS["mcp"],
        timeout_status=Status.ERROR,
        requires=(),
    )


def derive_mcp_servers(
    settings: HealthSettings, expanded: Mapping[str, Any] | None
) -> CategoryRecord | None:
    """Derive the auto ``mcp_servers`` health category from the config.

    Args:
        settings: Parsed ``health:`` settings; ``settings.auto`` gates the
            derivation and selects the target server URL.
        expanded: The ``${VAR}``-resolved config mapping (or ``None`` when no
            usable config loaded).

    Returns:
        ``None`` when ``health.auto.mcp.enabled`` is false. Otherwise a poll-cost
        :class:`CategoryRecord` named ``mcp_servers`` holding one
        :class:`CheckSpec` of ``type="mcp"`` per qualifying
        ``claude_code.servers`` entry — possibly zero checks when the config is
        absent, empty, or has no qualifying servers (the empty category is valid
        and returned).
    """
    if not settings.auto.enabled:
        return None

    url_key = _resolve_url_key(settings)
    budget = resolve_callable_timeout_s(Cost.POLL, None, settings.suite_timeout_s)

    checks: list[CheckSpec] = []
    servers = _servers_mapping(expanded)
    for raw_name, entry in servers.items():
        check = _check_for_entry(str(raw_name), entry, url_key)
        if check is not None:
            checks.append(check)

    return CategoryRecord(name=_CATEGORY_NAME, cost=Cost.POLL, timeout_s=budget, checks=checks)


def _servers_mapping(expanded: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return ``claude_code.servers`` as a mapping, or an empty one if absent.

    Every missing or wrong-typed level (``expanded`` ``None``, no ``claude_code``,
    a non-mapping ``claude_code`` or ``servers``) collapses to ``{}`` so the
    caller derives a zero-check category rather than raising.
    """
    if not isinstance(expanded, Mapping):
        return {}
    claude_code = expanded.get("claude_code")
    if not isinstance(claude_code, Mapping):
        return {}
    servers = claude_code.get("servers")
    if not isinstance(servers, Mapping):
        return {}
    return servers
