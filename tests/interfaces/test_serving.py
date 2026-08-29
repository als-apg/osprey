"""Contract test for :mod:`osprey.interfaces._serving`.

Pins ``authorize_browser_context`` to mint an *ephemeral* (unpersisted)
session: the module's whole reason for existing is authorizing a browser
against a server this same process just spun up, so a session that leaked
into a persisted store would go on outliving the throwaway server it was
minted for.
"""

from __future__ import annotations

from osprey.interfaces._serving import authorize_browser_context
from osprey.interfaces.common_middleware import session_cookie_name
from osprey.interfaces.web_auth import _digest, get_web_credentials


class _FakeContext:
    """A stand-in exposing only the ``add_cookies`` method the helper calls."""

    def __init__(self) -> None:
        self.cookies: list[dict[str, str]] = []

    def add_cookies(self, cookies: list[dict[str, str]]) -> None:
        self.cookies.extend(cookies)


def test_authorize_browser_context_mints_an_ephemeral_session():
    context = _FakeContext()

    session_id = authorize_browser_context(context)

    assert _digest(session_id) in get_web_credentials()._ephemeral
    assert context.cookies == [
        {
            "name": session_cookie_name(),
            "value": session_id,
            "domain": "127.0.0.1",
            "path": "/",
        }
    ]
    assert get_web_credentials().verify_session(session_id) is True
