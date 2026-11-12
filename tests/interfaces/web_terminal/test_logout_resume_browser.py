"""Browser test: multi-user logout -> landing -> return resumes the server's last-active.

Exercises the client-side half of the multi-user round trip in a real Chromium
page -- the part a FastAPI TestClient can't see because it neither runs the
frontend JS nor persists ``localStorage`` across navigations:

  * the More menu's identity line naming the user and the sidebar footer's
    Sign out row (``#logout-btn`` carrying ``data-landing-url``) render only
    when the server emitted a non-empty ``terminal_user`` / ``landing_url``
    (multi-user);
  * clicking Sign out asks first (the sign-out dialog); confirming POSTs the
    real server logout route (``logout_terminal``, routes/websocket.py),
    clears the client's stored pointer, THEN navigates to the landing origin;
  * the ruling: logout clears the browser's pointer, and a later login resumes
    the session the SERVER recorded as last active (``session-meta.json``'s
    ``last_active`` outlives ``logout_terminal``). With the server's answer
    withheld, the return opens a fresh station -- the proof the pointer is gone;
    with it, the return resumes that key (``mode=resume``);
  * the pointer is the PER-RENDER key. On a multi-user mount every ``/u/<user>/``
    shares one origin and so one ``localStorage``; the server stamps
    ``data-osprey-storage-scope`` on ``<html>`` and the client reads, writes and
    clears ``osprey-pty-session--<user>``, never the bare shared slot.

Scope note -- no live model turn. A warm session is represented by seeding the
pointer and the server's last-active directly; the shell is a long-lived
``sleep``, so a resume connection stays open rather than masking the
difference.

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from osprey.interfaces._serving import authorize_browser_context
from tests.interfaces._browser import wait_for_page_ready
from tests.interfaces.web_terminal._surface import arm_terminal_tab
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


def _put_last_active(base_url: str, key: str) -> None:
    """Record ``key`` as the operator's last-active session, as the page does."""
    import httpx

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
    from osprey.interfaces.web_auth import get_web_credentials

    response = httpx.put(
        f"{base_url}/api/sessions/last-active",
        json={"session_key": key},
        headers={OPERATOR_SECRET_HEADER: get_web_credentials().operator_secret},
        timeout=10.0,
    )
    assert response.status_code == 200, response.text


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_logout_clears_the_pointer_and_login_resumes_the_server_last_active(
    tmp_path, monkeypatch, chromium_browser
):
    """Logout empties the scoped pointer; the next login resumes the server's last-active."""
    user = "operator-alpha"
    stored_id = "11111111-2222-3333-4444-555555555555"

    with (
        launch_landing() as landing_url,
        launch_terminal(
            tmp_path, monkeypatch, terminal_user=user, landing_url=landing_url
        ) as base_url,
    ):
        page = chromium_browser.new_page()
        arm_terminal_tab(page, scope=user)

        # --- First load: multi-user chrome renders ---
        with page.expect_websocket() as first_ws:
            page.goto(base_url, wait_until="load")
        assert "mode=resume" not in first_ws.value.url
        wait_for_page_ready(page)

        logout = page.locator("#logout-btn")
        expect(logout).to_have_count(1)
        assert logout.get_attribute("data-landing-url") == landing_url
        expect(logout).to_be_visible()

        # --- A warm session: the scoped pointer and the server's last-active ---
        scope = page.evaluate(
            "() => document.documentElement.getAttribute('data-osprey-storage-scope')"
        )
        assert scope == user, f"multi-user page must be stamped with its render, got {scope!r}"
        pointer_key = f"osprey-pty-session--{scope}"
        # Let the page's own debounced last-active write settle before ours.
        page.wait_for_timeout(600)
        page.evaluate("([k, id]) => localStorage.setItem(k, id)", [pointer_key, stored_id])
        _put_last_active(base_url, stored_id)

        auth_requests: list[str] = []
        page.on(
            "request",
            lambda request: (
                auth_requests.append(request.url) if "/auth/logout" in request.url else None
            ),
        )

        # --- Logout: asks first, then lands ---
        page.click("#more-btn")
        expect(page.locator("#more-menu .more-menu-user")).to_have_text(user)
        page.keyboard.press("Escape")
        page.click("#logout-btn")
        dialog = page.locator(".posture-modal-overlay:not([data-closing])")
        expect(dialog.locator(".posture-modal-title")).to_have_text("Sign out of OSPREY?")
        dialog.locator(".posture-modal-confirm").click()
        # Sign-out ends the server's sessions before it navigates, which a
        # loaded host does slowly: the wait is for the navigation to commit,
        # with no budget of its own (0; the per-test timeout is the hang
        # breaker), and readiness is the landing stand-in's own marker.
        page.wait_for_url(
            lambda u: u.startswith(landing_url),
            wait_until="commit",
            timeout=0,
        )  # page-ready: the landing stand-in is this test's own page
        expect(page.locator("#landing-marker")).to_be_visible()

        # The auth session was asked to end at the ORIGIN ROOT, naming this user once.
        assert len(auth_requests) == 1, auth_requests
        requested = urlsplit(auth_requests[0])
        assert requested.path == "/auth/logout"
        assert parse_qs(requested.query) == {"user": [user]}

        # Logout revoked the cookie; the perimeter re-issues one on return.
        authorize_browser_context(page.context)

        # --- Return with the server's answer withheld: the pointer is gone ---
        page.route("**/api/sessions/last-active", lambda route: route.abort())
        with page.expect_websocket() as cut_ws:
            page.goto(base_url, wait_until="load")
            wait_for_page_ready(page)
        assert f"session_id={stored_id}" not in cut_ws.value.url, (
            f"logout left the stored pointer behind: {cut_ws.value.url}"
        )
        assert page.evaluate("(k) => localStorage.getItem(k)", pointer_key) != stored_id
        assert page.evaluate("() => localStorage.getItem('osprey-pty-session')") is None
        page.unroute("**/api/sessions/last-active")

        # --- A later login: the server's last-active is resumed ---
        page.wait_for_timeout(600)
        _put_last_active(base_url, stored_id)
        with page.expect_websocket() as return_ws:
            page.goto(base_url, wait_until="load")
            wait_for_page_ready(page)
        resumed = return_ws.value.url
        # The socket asks for the server's key. (No PTY by that id lives in this
        # stand-in, so what the client stores after the server's answer is not
        # the point here.)
        assert "mode=resume" in resumed and f"session_id={stored_id}" in resumed, resumed
        assert page.evaluate("() => localStorage.getItem('osprey-pty-session')") is None

        page.close()
