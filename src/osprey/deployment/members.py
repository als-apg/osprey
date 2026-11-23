"""The uv workspace members that ship beside the framework, and those that do not.

Each name is both the member's directory under ``packages/`` and its
distribution name. The deployment plumbing reads ``WORKSPACE_MEMBERS`` rather
than the monorepo's root ``pyproject.toml``, so an installed framework that has
no checkout beside it still knows which member wheels it needs.
``NOT_SHIPPED`` names the members the framework does not depend on: they are
developed in the same workspace but stand alone, so the plumbing never builds
or installs them with the framework. A test pins ``WORKSPACE_MEMBERS +
NOT_SHIPPED`` equal to the root ``[tool.uv.workspace].members``, in order.
"""

from __future__ import annotations

WORKSPACE_MEMBERS: tuple[str, ...] = (
    "osprey-connectors",
    "osprey-facility",
    "osprey-optimizer",
    "pyaml-cs-osprey",
)

NOT_SHIPPED: tuple[str, ...] = ("osprey-panels",)
