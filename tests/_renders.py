"""Builders for the render projection a rendered config carries.

A built deployment states its logins under ``modules.web_terminals.renders``,
one entry per render, each tagged with its ``kind`` (always ``tier``: a shared
login is a login file in ``tiers/`` too). Tests that hand a rendered reader a
config spell that catalog through this builder, so every test speaks the one
shape the build writes::

    >>> tier_render("assistant", "demo-assistant")
    {'kind': 'tier', 'project': 'demo-assistant'}

``name`` is the render's catalog key. It is not stored in the entry (the key
already carries it); it is spelled at the call site so a ``renders`` literal
reads ``{"assistant": tier_render("assistant", ...)}``. A ``project`` of
``None`` leaves the key out, for the tests that judge an entry missing it.
"""

from __future__ import annotations

from typing import Any

from osprey.deployment.web_terminals.render_naming import RENDER_KIND_TIER

__all__ = ["tier_render"]


def _render(kind: str, project: str | None, project_path: str | None, knobs: Any) -> dict:
    entry: dict[str, Any] = {"kind": kind}
    if project is not None:
        entry["project"] = project
    if project_path is not None:
        entry["project_path"] = project_path
    entry.update(knobs)
    return entry


def tier_render(
    name: str, project: str | None, *, project_path: str | None = None, **knobs: Any
) -> dict[str, Any]:
    """One ``renders`` entry of kind ``tier``: ``{"kind": "tier", "project": …, **knobs}``."""
    del name
    return _render(RENDER_KIND_TIER, project, project_path, knobs)
