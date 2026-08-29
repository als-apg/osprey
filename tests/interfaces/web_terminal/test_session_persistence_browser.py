"""Browser tests: the login cookie outlives the browser window, and logout ends it.

The token exchange stamps its ``Set-Cookie`` with ``Max-Age`` (the configured
session lifetime) rather than leaving it a browser-session cookie, so an
operator who quits the control-room browser and reopens it is still logged in.
That claim is about the *browser's* half of the credential — whether Chromium
writes the cookie to disk and offers it again on a cold start — and no
``TestClient`` can answer it: a test client reads a header, it does not run a
cookie jar with a persistence rule. So this drives a real Chromium.

What each test proves:

  * **The exchange hands out a persistent cookie.** After a real ``?token=``
    login, the context holds ``osprey_terminal_session*`` with an ``expires``
    that is a wall-clock deadline (not Playwright's ``-1``, which is how a
    session cookie is spelled), and that deadline matches the lifetime the
    *server* will honour — the two expiries agreeing is the point of stamping
    the number rather than picking one.
  * **A browser restart keeps it.** The context's storage state is filtered
    down to the cookies a browser actually carries across a quit — the ones
    with a real expiry — and a fresh context is opened from that. It loads the
    terminal with no ``?token=`` in the URL and renders.
  * **Logout ends it on both sides.** Clicking Log out lands the browser on the
    landing page, the cookie is gone from the jar (the ``Max-Age=0`` delete
    landed), and the terminal refuses a return visit with ``401`` — including
    when the recorded cookie value is put *back* in the jar by hand, which is
    what makes it a server-side revocation rather than a client-side tidy-up.

**Why the fixture's authorization is deliberately thrown away.** ``chromium_browser``
wraps ``new_context``/``new_page`` so every context is born holding a session
minted in this process (``authorize_browser_context``, ``persist=False``) — the
seam that lets the other browser suites skip the login entirely. That session is
*ephemeral*, and it is installed through ``add_cookies`` with no expiry, so a
test that let it authenticate would be asserting the persistence of a cookie the
exchange never issued. Both tests below therefore clear the jar before their
first navigation and log in for real; where a context is rebuilt from storage
state, the seam's freshly-minted cookie is cleared again and the recorded one
re-installed by value, so what authenticates is provably the exchange's copy.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pytest

from osprey.interfaces.common_middleware import session_cookie_name
from osprey.interfaces.web_auth import get_web_credentials
from tests.interfaces.web_terminal.test_logout_resume_browser import (
    _PLAYWRIGHT_AVAILABLE,
    _launch_landing,
    _launch_terminal,
)

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: How far the browser's stamped expiry may sit from the server's own deadline
#: before the two count as disagreeing. Generous because it absorbs the whole
#: round trip — app startup, the navigation, Chromium writing the jar — and the
#: failure this guards against is an expiry off by hours or set to zero, never
#: one off by a minute.
_EXPIRY_TOLERANCE_SECONDS = 120.0

#: The terminal shell's first-paint marker, the same one the sibling browser
#: suites wait on. Present exactly when the app rendered rather than refused.
_TERMINAL_MARKER = ".header-actions"


def _session_cookies(context) -> list[dict]:
    """Every cookie in *context* carrying the gate's session name.

    The name is derived here the way :class:`WebAuthMiddleware` derives the one
    it looks for — ``session_cookie_name()``, off ``OSPREY_WEB_PORT`` — so a
    deployment that port-suffixes the name cannot make this suite quietly assert
    on a cookie nobody sets.
    """
    name = session_cookie_name()
    return [cookie for cookie in context.cookies() if cookie["name"] == name]


def _log_in(context, base_url: str) -> dict:
    """Trade the operator secret for a session cookie in a real browser.

    Clears the jar first: the ``chromium_browser`` seam has already put an
    ephemeral session in it, and leaving that behind would let the page render
    on a credential this test is not making a claim about.

    Returns:
        The session cookie the exchange set, as Playwright reports it.
    """
    context.clear_cookies()
    assert _session_cookies(context) == [], "the fixture's seam cookie was not cleared"

    secret = get_web_credentials().operator_secret
    page = context.new_page()
    page.goto(f"{base_url}/?token={quote(secret, safe='')}", wait_until="load")
    page.wait_for_selector(_TERMINAL_MARKER, timeout=10_000)
    # The secret is spent and gone from the address bar; the cookie took over.
    assert "token=" not in page.url

    cookies = _session_cookies(context)
    assert len(cookies) == 1, f"expected exactly one session cookie, got {cookies}"
    return cookies[0]


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_login_cookie_survives_a_browser_restart(tmp_path, monkeypatch, chromium_browser):
    """A cold browser start within the lifetime is still logged in.

    The exchange's cookie carries a real ``Max-Age``, so Chromium keeps it on
    disk rather than dropping it with the window. A restart is modelled the way
    a browser performs one: the storage state is taken, every cookie *without* a
    wall-clock expiry is discarded from it — that is precisely what a quit does
    — and a brand-new context is built from what is left. The terminal then
    loads with no token anywhere in the URL.
    """
    with _launch_terminal(tmp_path, monkeypatch) as base_url:
        context = chromium_browser.new_context()
        try:
            cookie = _log_in(context, base_url)

            # The browser was given a deadline, not a "until you close me".
            assert cookie["expires"] != -1, "exchange handed out a browser-session cookie"
            ttl = get_web_credentials().session_ttl_seconds
            drift = abs(cookie["expires"] - (time.time() + ttl))
            assert drift < _EXPIRY_TOLERANCE_SECONDS, (
                f"cookie expiry disagrees with the server's {ttl}s session lifetime by {drift}s"
            )
            # ...and it is the cookie the gate will actually be offered.
            assert cookie["path"] == "/"
            assert cookie["httpOnly"] is True

            state = context.storage_state()
            surviving = [c for c in state["cookies"] if c.get("expires", -1) != -1]
            assert any(c["name"] == cookie["name"] for c in surviving), (
                "the session cookie is not among the cookies a browser restart keeps"
            )
        finally:
            context.close()

        # --- The restart: a new context holding only what survived the quit ---
        restarted = chromium_browser.new_context(storage_state={**state, "cookies": surviving})
        try:
            # ``new_context`` minted a fresh ephemeral session through the
            # fixture's seam and installed it under the same name, path and
            # domain — which overwrites the restored cookie. Put the restored
            # one back by value so the navigation below is authenticated by the
            # exchange's copy and by nothing else.
            restarted.clear_cookies()
            restarted.add_cookies([c for c in surviving if c["name"] == cookie["name"]])
            assert [c["value"] for c in _session_cookies(restarted)] == [cookie["value"]]

            page = restarted.new_page()
            response = page.goto(base_url, wait_until="load")
            assert response is not None
            assert response.status == 200, "the restored cookie was not accepted"
            page.wait_for_selector(_TERMINAL_MARKER, timeout=10_000)
            assert "token=" not in page.url
        finally:
            restarted.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_logout_ends_the_persistent_session_on_both_sides(tmp_path, monkeypatch, chromium_browser):
    """Log out deletes the browser's copy and revokes the server's.

    Run in the multi-user shape, because that is where logout has a landing page
    to send the operator to and where the delete cookie has to cross the
    ``/u/<user>/`` prefix (it is ``Path=/``, so it does). Two refusals are
    asserted, and the second is the one that matters: putting the recorded
    cookie value back in the jar by hand still gets a ``401``, so the session is
    gone from the server, not merely from the browser.
    """
    user = "alice"
    with (
        _launch_landing() as landing_url,
        _launch_terminal(
            tmp_path, monkeypatch, terminal_user=user, landing_url=landing_url
        ) as base_url,
    ):
        context = chromium_browser.new_context()
        try:
            cookie = _log_in(context, base_url)
            assert cookie["expires"] != -1, (
                "the delete must clear a cookie the browser wrote to disk"
            )

            page = context.pages[0]

            # --- Log out, from where an operator reaches for it ---
            page.click("#header-identity-trigger")
            page.click("#logout-btn")
            page.wait_for_url(lambda url: url.startswith(landing_url))
            page.wait_for_selector("#landing-marker", timeout=10_000)

            # The ``Max-Age=0`` delete landed despite the prefix.
            assert _session_cookies(context) == [], "logout left the session cookie in the jar"

            # --- The return visit is refused ---
            response = page.goto(base_url, wait_until="domcontentloaded")
            assert response is not None
            assert response.status == 401
            assert page.locator(_TERMINAL_MARKER).count() == 0

            # --- ...and still refused with the old cookie handed back ---
            context.add_cookies([cookie])
            assert [c["value"] for c in _session_cookies(context)] == [cookie["value"]]
            response = page.goto(base_url, wait_until="domcontentloaded")
            assert response is not None
            assert response.status == 401, "the revoked session was still accepted"
            assert page.locator(_TERMINAL_MARKER).count() == 0
        finally:
            context.close()
