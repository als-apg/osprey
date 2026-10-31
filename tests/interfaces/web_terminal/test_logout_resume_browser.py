"""Browser test: multi-user logout -> landing -> return starts a FRESH station.

Exercises the client-side half of the multi-user round trip in a real Chromium
page — the part a FastAPI TestClient can't see because it neither runs the
frontend JS nor persists ``localStorage`` across navigations:

  * the ☰ More menu's identity line naming the user and the sidebar footer's
    Sign out row (``#logout-btn`` carrying ``data-landing-url``) render only
    when the server emitted a non-empty ``terminal_user`` / ``landing_url``
    (multi-user);
  * clicking Sign out asks first (the sign-out dialog); confirming POSTs the
    real server logout route (``logout_terminal``, routes/websocket.py —
    empties the PTY and operator registries), clears the client's stored PTY
    session id, THEN navigates to the configured landing origin;
  * returning to the terminal origin does NOT resume the prior warm PTY —
    this render's stored pointer stays empty across the round trip and the
    client opens a brand-new WebSocket (no ``mode=resume``, no stale
    ``session_id``) rather than reconnecting to the old session. This is M2:
    logout is a real station reset, not just a client-side navigation.
  * the pointer is the PER-RENDER key. On a multi-user mount every ``/u/<user>/``
    shares one origin and so one ``localStorage``; the server stamps
    ``data-osprey-storage-scope`` on ``<html>`` and terminal.js reads, writes
    and clears ``osprey-pty-session--<user>`` (design_system/storage-scope.js),
    never the bare shared slot. This is the only multi-user browser suite, so it
    is also the live proof that the scoped clear is what logout performs.

Scope note — no live model turn. A genuinely live PTY session id is minted by
Claude (``SessionDiscovery`` watching for the CLI's ``.jsonl`` file), which needs
the ``claude`` binary + a provider and is infeasible in a headless CI browser.
So this asserts the reset *mechanism* deterministically: the shell command is a
long-lived ``sleep`` (so a supposed resume connection would stay open rather
than exiting, which would otherwise mask a resume-vs-fresh distinction), and a
warm session is represented by seeding ``localStorage`` directly before logout.
What's proven is the client contract — the stored pointer is gone and the
post-logout WebSocket asks for a fresh session — not a real Claude conversation
being torn down.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from osprey.interfaces._serving import authorize_browser_context
from tests.interfaces.web_terminal._user_terminal import launch_landing, launch_terminal

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_logout_and_return_starts_fresh_session(tmp_path, monkeypatch, chromium_browser):
    """Full multi-user loop: identity + Sign out render, logout resets the station.

    Closes M2: a warm PTY (represented here by a seeded ``localStorage`` id, see
    the module docstring) does NOT survive the logout -> landing -> return round
    trip. Logout clears the client's stored session id before navigating away
    (logout.js, run by the sign-out dialog; backed server-side by the real
    ``logout_terminal`` registry cleanup), so the return visit opens a brand-new
    terminal — no ``mode=resume``, no ``session_id`` carried over from the prior
    visit, and no re-adopted stored id — rather than reconnecting to the old
    warm PTY the earlier (Phase-1) version of this test exercised.
    """
    user = "operator-alpha"
    stored_id = "11111111-2222-3333-4444-555555555555"

    with (
        launch_landing() as landing_url,
        launch_terminal(
            tmp_path, monkeypatch, terminal_user=user, landing_url=landing_url
        ) as base_url,
    ):
        page = chromium_browser.new_page()

        # --- First load: multi-user chrome renders, fresh (new) session opens ---
        with page.expect_websocket() as first_ws:
            page.goto(base_url, wait_until="load")
        opening_url = first_ws.value.url
        assert "mode=resume" not in opening_url, (
            f"first visit with empty storage should open a NEW session, got {opening_url}"
        )
        assert "session_id=" not in opening_url

        # The logout control carries the landing url.
        logout = page.locator("#logout-btn")
        expect(logout).to_have_count(1)
        assert logout.get_attribute("data-landing-url") == landing_url
        # It is a row of the sidebar footer, on screen from the first paint.
        expect(logout).to_be_visible()

        # --- Represent an established (warm) PTY session (see module docstring) ---
        # Seed the key the client actually reads: derived from the scope the
        # server stamped for this render, exactly as storage-scope.js derives it.
        scope = page.evaluate(
            "() => document.documentElement.getAttribute('data-osprey-storage-scope')"
        )
        assert scope == user, f"multi-user page must be stamped with its render, got {scope!r}"
        session_key = f"osprey-pty-session--{scope}"
        page.evaluate("([k, id]) => localStorage.setItem(k, id)", [session_key, stored_id])
        captured = page.evaluate("(k) => localStorage.getItem(k)", session_key)
        assert captured == stored_id

        # Record the auth-sidecar chaining request. This deployment has NO
        # sidecar (no auth stanza, so nginx renders no `location /auth/` and
        # nothing is listening), which makes this the live proof of the
        # authentication-off case: the request is made, it fails, and the
        # logout still completes. A *navigation* to the sidecar here would
        # strand the browser on a 404 instead of the landing page.
        auth_requests: list[str] = []

        def _record(request) -> None:
            if "/auth/logout" in request.url:
                auth_requests.append(request.url)

        page.on("request", _record)

        # --- Logout: clears the stored pointer, THEN navigates to landing ---
        # The ☰ More menu names the operator, so they can confirm WHICH terminal
        # they are leaving before they leave it.
        page.click("#more-btn")
        expect(page.locator("#more-menu .more-menu-user")).to_have_text(user)
        page.keyboard.press("Escape")
        # Sign out is the sidebar footer's row, and it asks first.
        page.click("#logout-btn")
        dialog = page.locator(".posture-modal-overlay:not([data-closing])")
        expect(dialog.locator(".posture-modal-title")).to_have_text("Sign out of OSPREY?")
        dialog.locator(".posture-modal-confirm").click()
        page.wait_for_url(lambda u: u.startswith(landing_url))
        assert page.url.startswith(landing_url)
        expect(page.locator("#landing-marker")).to_be_visible()

        # The auth session was asked to end, addressed at the ORIGIN ROOT — not
        # under `/u/<user>/` (which `base_url` is, and which would reach this
        # container rather than the sidecar) — and naming exactly this user once.
        assert auth_requests, "logout did not attempt to end the auth session"
        assert len(auth_requests) == 1
        requested = urlsplit(auth_requests[0])
        assert requested.path == "/auth/logout"
        assert parse_qs(requested.query) == {"user": [user]}

        # NOTE: we can't read the terminal origin's localStorage from here —
        # the page has navigated to the landing origin (a different
        # scheme+host+port), and localStorage is origin-scoped. Whether
        # clearStoredSessionId() actually ran is only observable once we're
        # back on the terminal origin below.

        # --- Return to the terminal origin: a FRESH station, not a resume ---
        # Logout revoked the browser session server-side and expired its cookie
        # (``logout_terminal``, routes/websocket.py), which is the point: the
        # credential must not survive the round trip. So the return visit needs
        # a NEW one, exactly as a returning operator gets from the perimeter
        # (nginx + the auth sidecar) before nginx ever proxies them back to this
        # container. ``chromium_browser``'s seam mints the first session at
        # ``new_page()`` and cannot re-mint mid-test, so the perimeter's re-issue
        # is stood in for here. Without it the navigation below is refused 401
        # and never opens a socket — which would say nothing about resume.
        authorize_browser_context(page.context)

        with page.expect_websocket() as return_ws:
            page.goto(base_url, wait_until="load")
        fresh_url = return_ws.value.url
        assert "mode=resume" not in fresh_url, (
            f"return visit after logout must start a fresh session, got {fresh_url}"
        )
        assert f"session_id={stored_id}" not in fresh_url

        # No re-adoption of the old id: the prior warm PTY is not inherited.
        assert page.evaluate("(k) => localStorage.getItem(k)", session_key) != stored_id
        # And a scoped page never touches the shared slot — neither the seed nor
        # the fresh session's confirmation lands under the bare key.
        assert page.evaluate("() => localStorage.getItem('osprey-pty-session')") is None

        page.close()
