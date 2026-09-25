"""Every registered panel tool is named on both pages that list the panel tools.

The panel tools are a code-level set: the workspace MCP server registers them
with ``@mcp.tool()`` in one module, and that set changes when a tool is added or
renamed. Two documentation pages enumerate it by hand — the operator's panels
how-to and the per-server tool inventory in the architecture section. A list
kept by hand in two places drifts without a sound: a new tool ships and the
pages keep describing the old surface, so the reader never learns it exists.

This guard binds each page to the code, not the pages to each other. The
registered names are read from the module's syntax tree rather than imported:
a docs guard that imports an MCP server module pays for that server's whole
import graph to check a list of names.

The check is a subset one. A page may name a tool in prose as often as it
likes; a renamed tool fails on the missing new name, which is the same edit
that removes the stale one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TOOL_MODULE = "src/osprey/mcp_server/workspace/tools/panel_tools.py"

_PAGES = (
    "docs/source/how-to/web-terminal/panels.rst",
    "docs/source/architecture/mcp-servers.rst",
)


def _is_mcp_tool_decorator(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tool"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mcp"
    )


def _registered_panel_tools() -> set[str]:
    tree = ast.parse((_REPO_ROOT / _TOOL_MODULE).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(_is_mcp_tool_decorator(d) for d in node.decorator_list)
    }


def _literals(page: str) -> set[str]:
    text = (_REPO_ROOT / page).read_text(encoding="utf-8")
    return set(re.findall(r"``([a-z_]+)``", text))


def test_the_panel_tool_decorator_is_still_recognised():
    registered = _registered_panel_tools()
    assert {"list_panels", "arrange_workspace"} <= registered, (
        f"{_TOOL_MODULE} no longer registers panel tools the way this guard reads "
        f"them; found {sorted(registered)}"
    )


@pytest.mark.parametrize("page", _PAGES)
def test_every_panel_tool_is_named_on_the_page(page):
    missing = _registered_panel_tools() - _literals(page)
    assert not missing, f"{page} does not name the panel tools {sorted(missing)}"
