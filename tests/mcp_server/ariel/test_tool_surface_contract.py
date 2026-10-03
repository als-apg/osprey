"""The ARIEL tool surface is documented and classified wherever it is named.

The live tool names come from the same walker the classification drift test
uses — the FastMCP singleton after every tool module is imported, with its
``list_tools`` middleware run — and never from ``create_server()``. Each name
must then appear in the three places a reader or the permission layer looks:

- ``docs/source/reference/contracts/ariel.rst`` — the tool contract table;
- ``CLAUDE.ariel.md.j2`` — the agent's tool surface, read as raw template text
  so a row inside a ``{% if %}`` still counts;
- ``FRAMEWORK_SERVERS["ariel"].permissions_allow`` — for every read tool, that
  is every tool not routed through ``permissions_ask``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS
from tests.registry.test_tool_classification_drift import _registered_tool_names

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_RST = _REPO_ROOT / "docs" / "source" / "reference" / "contracts" / "ariel.rst"
_MCP_SERVERS_RST = _REPO_ROOT / "docs" / "source" / "architecture" / "mcp-servers.rst"
_TEMPLATE = _REPO_ROOT / "src" / "osprey" / "templates" / "claude_code" / "CLAUDE.ariel.md.j2"

_ARIEL = FRAMEWORK_SERVERS["ariel"]


@pytest.fixture(scope="module")
def live_tools() -> set[str]:
    return _registered_tool_names(_ARIEL.module)


def _rst_literals(path: Path) -> set[str]:
    return set(re.findall(r"``([a-z_]+)``", path.read_text(encoding="utf-8")))


def _md_literals(path: Path) -> set[str]:
    return set(re.findall(r"(?<!`)`([a-z_]+)`(?!`)", path.read_text(encoding="utf-8")))


def test_walker_sees_attachment_view(live_tools):
    """The default-on view switch leaves ``attachment_view`` in the walked surface."""
    assert "attachment_view" in live_tools
    assert "hybrid_search" in live_tools


def test_every_live_tool_is_in_the_contract_table(live_tools):
    missing = live_tools - _rst_literals(_CONTRACT_RST)
    assert not missing, f"contracts/ariel.rst does not name: {sorted(missing)}"


def test_every_live_tool_is_in_the_mcp_servers_page(live_tools):
    missing = live_tools - _rst_literals(_MCP_SERVERS_RST)
    assert not missing, f"architecture/mcp-servers.rst does not name: {sorted(missing)}"


def test_every_live_tool_is_in_the_agent_template(live_tools):
    missing = live_tools - _md_literals(_TEMPLATE)
    assert not missing, f"CLAUDE.ariel.md.j2 does not name: {sorted(missing)}"


def test_every_read_tool_is_allowed_without_asking(live_tools):
    read_tools = live_tools - set(_ARIEL.permissions_ask)
    missing = read_tools - set(_ARIEL.permissions_allow)
    assert not missing, f"read tools missing from permissions_allow: {sorted(missing)}"


def test_agent_template_carries_no_literal_tool_count():
    """A count goes stale the moment a tool is added; the list itself is the count."""
    text = _TEMPLATE.read_text(encoding="utf-8")
    assert not re.search(r"\b\d+\s+ARIEL\s+MCP\s+tools\b", text)


def test_attachment_view_row_is_conditional_on_the_view_switch():
    text = _TEMPLATE.read_text(encoding="utf-8")
    row = text.index("- `attachment_view`")
    opener = text.rfind("{%- if ariel_attachment_view %}", 0, row)
    assert opener != -1
    assert "{%- endif %}" in text[row:]
    assert "{%- endif %}" not in text[opener:row]


def test_contract_documents_the_view_switch():
    """The view-off rule is stated in the contract next to ``attachment_view``."""
    text = _CONTRACT_RST.read_text(encoding="utf-8")
    assert "ariel.attachments.view.enabled" in text
    for key in ("attachment_count", "matched_attachment_ids", "viewable"):
        assert f"``{key}``" in text
