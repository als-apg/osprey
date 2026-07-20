"""Browser test: multi-user logout -> landing -> return starts a FRESH station.

Exercises the client-side half of the multi-user round trip in a real Chromium
page — the part a FastAPI TestClient can't see because it neither runs the
frontend JS nor persists ``localStorage`` across navigations:

  * the header user display (``#header-user-name``) and the logout control
    (``#logout-btn`` carrying ``data-landing-url``) render only when the server
    emitted a non-empty ``terminal_user`` / ``landing_url`` (multi-user);
  * clicking logout POSTs the real server logout route (``logout_terminal``,
    routes/websocket.py — empties the PTY and operator registries), clears the
    client's stored PTY session id, THEN navigates to the configured landing
    origin;
  * returning to the terminal origin does NOT resume the prior warm PTY —
    ``localStorage['osprey-pty-session']`` stays empty across the round trip and
    the client opens a brand-new WebSocket (no ``mode=resume``, no stale
    ``session_id``) rather than reconnecting to the old session. This is M2:
    logout is a real station reset, not just a client-side navigation.
  * plain ``osprey web`` (no landing_url) omits the logout control entirely.

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

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.browser, pytest.mark.slow]

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False


# A PTY command that stays alive AND tolerates the ``--resume <id>`` (and any
# ``--effort <level>``) args the websocket route appends on a resume connection.
# ``sleep``/``cat``/``echo`` would each exit or error on the extra args, which
# would trip terminal.js's auto-resume failover and clear the stored id mid-test.
_LONG_LIVED_SHELL = [sys.executable, "-c", "import time; time.sleep(3600)"]


# ---------------------------------------------------------------------------
# Live-server context managers
# ---------------------------------------------------------------------------


def _nginx_stand_in(app, prefix: str):
    """ASGI wrapper reproducing nginx's prefix contract for a per-user app.

    In a real multi-user deployment the browser talks to nginx at
    ``/u/<user>/…`` and nginx strips that prefix before proxying, so the app
    only ever sees bare paths (see the ``root_path`` note in ``create_app``).
    This wrapper is that stripping proxy: http/websocket scopes whose path
    starts with the prefix are forwarded with the prefix removed; the
    ``lifespan`` scope passes through untouched so the inner app's startup
    still runs (a ``starlette`` ``Mount`` would swallow it).
    """

    async def asgi(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            if scope["path"] == prefix:  # bare /u/<user>, as nginx's trailing-slash redirect
                scope = dict(scope, path="/")
            elif scope["path"].startswith(f"{prefix}/"):
                scope = dict(scope, path=scope["path"][len(prefix) :])
        await app(scope, receive, send)

    return asgi


@contextmanager
def _launch_terminal(
    tmp_path, monkeypatch, *, terminal_user: str = "", landing_url: str = ""
) -> Iterator[str]:
    """Launch the web terminal with ``OSPREY_TERMINAL_{USER,LANDING_URL}`` set.

    The env vars are read by ``create_app`` (app.py:443-444) into ``app.state``,
    so they MUST be in ``os.environ`` before the factory runs. Companion backends
    (artifact server, panels) are bypassed via the same patch set the other
    web-terminal browser suites use.

    With a ``terminal_user`` the SPA self-addresses under ``/u/<user>/`` and the
    app serves bare paths, so the pair only functions behind a prefix-stripping
    proxy; the server is wrapped in ``_nginx_stand_in`` and the yielded base URL
    carries the prefix, exactly as a user would reach it through nginx.
    """
    monkeypatch.chdir(tmp_path)
    env = {
        "OSPREY_TERMINAL_USER": terminal_user,
        "OSPREY_TERMINAL_LANDING_URL": landing_url,
    }
    with (
        patch.dict(os.environ, env),
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            side_effect=lambda a: setattr(a.state, "artifact_server_url", "http://127.0.0.1:8086"),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=list(_LONG_LIVED_SHELL))
        if terminal_user:
            prefix = f"/u/{terminal_user}"
            with _run_app_server(_nginx_stand_in(app, prefix)) as server_url:
                yield f"{server_url}{prefix}"
        else:
            with _run_app_server(app) as base_url:
                yield base_url


@contextmanager
def _launch_landing() -> Iterator[str]:
    """A trivial second origin standing in for the operator landing page."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _root() -> str:
        return (
            "<!doctype html><html><body><h1 id='landing-marker'>OSPREY LANDING</h1></body></html>"
        )

    with _run_app_server(app) as base_url:
        yield base_url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_logout_and_return_starts_fresh_session(tmp_path, monkeypatch, chromium_browser):
    """Full multi-user loop: header + logout render, logout resets the station.

    Closes M2: a warm PTY (represented here by a seeded ``localStorage`` id, see
    the module docstring) does NOT survive the logout -> landing -> return round
    trip. Logout clears the client's stored session id before navigating away
    (task 4.2's ``initLogoutButton``, backed server-side by task 4.1's real
    ``logout_terminal`` registry cleanup), so the return visit opens a brand-new
    terminal — no ``mode=resume``, no ``session_id`` carried over from the prior
    visit, and no re-adopted stored id — rather than reconnecting to the old
    warm PTY the earlier (Phase-1) version of this test exercised.
    """
    user = "operator-alpha"
    stored_id = "11111111-2222-3333-4444-555555555555"

    with (
        _launch_landing() as landing_url,
        _launch_terminal(
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

        # Header shows the configured user; logout control carries the landing url.
        expect(page.locator("#header-user-name")).to_have_text(user)
        logout = page.locator("#logout-btn")
        expect(logout).to_have_count(1)
        assert logout.get_attribute("data-landing-url") == landing_url

        # --- Represent an established (warm) PTY session (see module docstring) ---
        page.evaluate("(id) => localStorage.setItem('osprey-pty-session', id)", stored_id)
        captured = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
        assert captured == stored_id

        # Dismiss the first-visit welcome overlay so it can't intercept the click
        # (it covers the header on a fresh server session).
        page.evaluate(
            "() => { const o = document.getElementById('welcome-overlay'); if (o) o.remove(); }"
        )

        # --- Logout: clears the stored pointer, THEN navigates to landing ---
        page.click("#logout-btn")
        page.wait_for_url(lambda u: u.startswith(landing_url))
        assert page.url.startswith(landing_url)
        expect(page.locator("#landing-marker")).to_be_visible()

        # NOTE: we can't read the terminal origin's localStorage from here —
        # the page has navigated to the landing origin (a different
        # scheme+host+port), and localStorage is origin-scoped. Whether
        # clearStoredSessionId() actually ran is only observable once we're
        # back on the terminal origin below.

        # --- Return to the terminal origin: a FRESH station, not a resume ---
        with page.expect_websocket() as return_ws:
            page.goto(base_url, wait_until="load")
        fresh_url = return_ws.value.url
        assert "mode=resume" not in fresh_url, (
            f"return visit after logout must start a fresh session, got {fresh_url}"
        )
        assert f"session_id={stored_id}" not in fresh_url

        # No re-adoption of the old id: the prior warm PTY is not inherited.
        assert page.evaluate("() => localStorage.getItem('osprey-pty-session')") != stored_id

        page.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_standalone_has_no_logout_control(tmp_path, monkeypatch, chromium_browser):
    """Plain ``osprey web`` (no landing_url env) omits the logout control.

    With neither ``OSPREY_TERMINAL_USER`` nor ``OSPREY_TERMINAL_LANDING_URL`` set,
    both the user display and the logout button must be absent from the DOM — the
    single-user experience is unchanged.
    """
    with _launch_terminal(tmp_path, monkeypatch, terminal_user="", landing_url="") as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="load")

        # The hub shell must have rendered before asserting an element's absence.
        page.wait_for_selector(".header-actions", timeout=10_000)

        expect(page.locator("#logout-btn")).to_have_count(0)
        expect(page.locator("#header-user-name")).to_have_count(0)

        page.close()
