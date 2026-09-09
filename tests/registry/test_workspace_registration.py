"""Every ``osprey_workspace`` tool is classified, and the list matches the package.

``permissions_allow`` and ``permissions_ask`` are hand-kept lists of tool short
names. Nothing errors when one drifts from what the package registers: a name
that no longer exists renders a rule for nothing, and a tool that reaches the
render without a rule falls through to a prompt. Both are silent, and both are a
posture change nobody chose.

So the vocabulary is pinned from both ends. Every registered tool is either
auto-allowed, approval-gated, or named in :data:`_DELIBERATELY_PROMPTED` with
the reason quoted from the registration; and every listed name is a tool the
package actually registers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS

_TOOLS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "osprey" / "mcp_server" / "workspace" / "tools"
)

#: The rail axis: the three verbs that change what an operator can launch at
#: all, rather than what is on screen right now. Quoted from the registration:
#: "remove_panel_from_rail costs them the ability to launch the panel back,
#: add_panel_to_rail puts an entry in front of them, and register_panel adds a
#: proxied upstream." ``setup_patch`` is gated for a different reason — it
#: rewrites the deployment's own config — and is therefore checked separately.
#:
#: ``register_panel`` belongs here for the strongest of the three reasons:
#: registering a proxied upstream is a network reach beyond the workspace, so it
#: is asked for even in a deployment that auto-allows the rest of the package.
_DELIBERATELY_PROMPTED = {"add_panel_to_rail", "remove_panel_from_rail", "register_panel"}


def _registered_tool_names() -> set[str]:
    """Short names of every ``@mcp.tool()`` under the workspace tools package."""
    names: set[str] = set()
    for path in sorted(_TOOLS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any("mcp.tool" in ast.unparse(dec) for dec in node.decorator_list):
                names.add(node.name)
    return names


@pytest.fixture(scope="module")
def sdef():
    assert "osprey_workspace" in FRAMEWORK_SERVERS
    return FRAMEWORK_SERVERS["osprey_workspace"]


@pytest.fixture(scope="module")
def registered() -> set[str]:
    tools = _registered_tool_names()
    assert tools, "the AST scan found no @mcp.tool() functions — the scan is broken"
    return tools


def test_every_registered_tool_is_classified(sdef, registered) -> None:
    """A new tool must be allowed or gated, never left to fall through."""
    classified = set(sdef.permissions_allow) | set(sdef.permissions_ask)
    assert registered <= classified


def test_no_listed_name_is_a_tool_the_package_does_not_have(sdef, registered) -> None:
    """A rule for a tool that no longer exists gates nothing."""
    assert set(sdef.permissions_allow) | set(sdef.permissions_ask) <= registered


def test_the_prompted_set_is_exactly_the_declared_one(sdef) -> None:
    """``permissions_ask`` holds the rail verbs plus the config editor, nothing else."""
    assert set(sdef.permissions_ask) == _DELIBERATELY_PROMPTED | {"setup_patch"}


def test_every_prompted_tool_has_its_approval_hook(sdef) -> None:
    """A tool in ``ask`` with no PreToolUse rule asks nothing at run time."""
    gated = {rule.matcher.removeprefix("mcp__osprey_workspace__") for rule in sdef.hooks_pre}
    assert gated == set(sdef.permissions_ask)
