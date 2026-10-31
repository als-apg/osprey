"""Minimal render contexts for the registry resolution and template suites."""

from __future__ import annotations

from typing import Any

from osprey.registry.mcp import resolve_servers
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR


def base_ctx(**overrides: Any) -> dict[str, Any]:
    """The two keys every server and agent resolution reads, plus ``overrides``."""
    ctx: dict[str, Any] = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
    }
    ctx.update(overrides)
    return ctx


def render_ctx(**overrides: Any) -> dict[str, Any]:
    """:func:`base_ctx` plus the keys the render funnel derives for every render.

    ``settings.json.j2`` concatenates ``agent_data_root`` into the ``Read()``
    globs, so a context without it renders filesystem-root grants that no real
    render produces; it renders ``cleanup_period_days`` unconditionally.
    """
    ctx = base_ctx(agent_data_root=DEFAULT_AGENT_DATA_BASE_DIR, cleanup_period_days=30)
    ctx.update(overrides)
    return ctx


def resolve_one(cfg: dict, name: str, ctx: dict | None = None) -> dict:
    """The one resolved server called ``name``, against :func:`render_ctx` by default."""
    servers = resolve_servers(cfg, ctx or render_ctx())
    matches = [s for s in servers if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} server, got {len(matches)}"
    return matches[0]
