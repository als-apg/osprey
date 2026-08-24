"""Registry ↔ frontend parity: every framework MCP server has a badge color.

``session-helpers.js`` keys its ``SERVER_COLORS`` map on the server names the
transcript reader emits — for framework servers, the registry names. A server
registered without a matching key silently renders the grey ``srv-unknown``
badge, which is how ``osprey_facility_knowledge``, ``phoebus``, ``bluesky``
and ``health`` were invisible on the activity view's color axis for months
(and how the ``workspace`` → ``osprey_workspace`` rename orphaned its color).

This test reads the JS source, so it fails the moment a server is added to
the registry without a color — the same pin pattern as the reader's
``test_captures_every_framework_server``.
"""

import re
from pathlib import Path

from osprey.registry.mcp import FRAMEWORK_SERVERS

_HELPERS_JS = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "web_terminal"
    / "static"
    / "js"
    / "session-helpers.js"
)

_CSS = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "web_terminal"
    / "static"
    / "css"
    / "session.css"
)


def _server_colors_map() -> dict[str, str]:
    """Parse the SERVER_COLORS object literal out of session-helpers.js."""
    source = _HELPERS_JS.read_text()
    match = re.search(r"const SERVER_COLORS = \{(.*?)\};", source, re.DOTALL)
    assert match, "SERVER_COLORS map not found in session-helpers.js"
    return dict(re.findall(r"['\"]?([\w-]+)['\"]?\s*:\s*'([\w-]+)'", match.group(1)))


def test_every_framework_server_has_a_badge_color():
    colors = _server_colors_map()
    registered = {definition.name for definition in FRAMEWORK_SERVERS.values()}
    missing = registered - colors.keys()
    assert not missing, f"framework servers without a SERVER_COLORS entry: {sorted(missing)}"


def test_every_badge_class_has_a_css_rule():
    css = _CSS.read_text()
    for server, css_class in _server_colors_map().items():
        assert f".{css_class}" in css, f"{server}: no .{css_class} rule in session.css"


def test_no_stale_color_keys():
    """A key naming no registered server is dead weight — the rename trap."""
    colors = _server_colors_map()
    registered = {definition.name for definition in FRAMEWORK_SERVERS.values()}
    stale = colors.keys() - registered
    assert not stale, f"SERVER_COLORS keys naming no registered server: {sorted(stale)}"
