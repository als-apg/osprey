"""Browser ↔ server parity: one grammar for a session key.

``activity-log-link.js`` decides which session id a link to the activity log
may carry, and :mod:`osprey.interfaces.web_terminal.session_key` decides which
keys a store will ever answer for. A link the browser builds from a key the
server would refuse names nothing, so the two must be one grammar.

The JS literal carries ``^``/``$`` and the Python pattern does not: the
anchors are the JS spelling of ``fullmatch``, since ``RegExp.test`` searches
while the server matches the whole string.
"""

import re
from pathlib import Path

from osprey.interfaces.web_terminal.session_key import _SESSION_KEY_RE

_LINK_JS = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "web_terminal"
    / "static"
    / "js"
    / "activity-log-link.js"
)


def _js_session_key_literal() -> str:
    """The body of ``const SESSION_KEY_RE = /…/;`` in activity-log-link.js."""
    source = _LINK_JS.read_text()
    match = re.search(r"const SESSION_KEY_RE = /(.+)/;", source)
    assert match, "SESSION_KEY_RE literal not found in activity-log-link.js"
    return match.group(1)


def test_browser_and_server_share_the_session_key_grammar():
    assert _js_session_key_literal() == f"^{_SESSION_KEY_RE.pattern}$"
