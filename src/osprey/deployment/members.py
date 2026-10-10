"""The uv workspace members that ship beside the framework.

Each name is both the member's directory under ``packages/`` and its
distribution name. The deployment plumbing reads this constant rather than the
monorepo's root ``pyproject.toml``, so an installed framework that has no
checkout beside it still knows which member wheels it needs. A test pins it
equal to the root ``[tool.uv.workspace].members``.
"""

from __future__ import annotations

WORKSPACE_MEMBERS: tuple[str, ...] = ("osprey-connectors", "pyaml-cs-osprey")
