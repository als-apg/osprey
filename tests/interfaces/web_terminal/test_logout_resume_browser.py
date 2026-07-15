"""Browser test: multi-user logout -> landing -> return resumes the SAME PTY.

Exercises the client-side half of the multi-user round trip in a real Chromium
page — the part a FastAPI TestClient can't see because it neither runs the
frontend JS nor persists ``localStorage`` across navigations:

  * the header user display (``#header-user-name``) and the logout control
    (``#logout-btn`` carrying ``data-landing-url``) render only when the server
    emitted a non-empty ``terminal_user`` / ``landing_url`` (multi-user);
  * clicking logout navigates the page to the configured landing origin;
  * returning to the terminal origin auto-resumes the *stored* PTY session —
    ``localStorage['osprey-pty-session']`` is unchanged AND the client opens its
    WebSocket with ``?session_id=<id>&mode=resume`` rather than a fresh session;
  * plain ``osprey web`` (no landing_url) omits the logout control entirely.

Scope note — no live model turn. A genuinely live PTY session id is minted by
Claude (``SessionDiscovery`` watching for the CLI's ``.jsonl`` file), which needs
the ``claude`` binary + a provider and is infeasible in a headless CI browser.
So this asserts the reconnect *mechanism* deterministically: the shell command
is a long-lived ``sleep`` (which tolerates the appended ``--resume <id>`` without
exiting, so the resume connection stays open and the one-shot failover never
clears the stored id), and the established session is represented by seeding
``localStorage`` directly. What's proven is the client contract — persistence
across the logout round trip and the exact resume WebSocket URL — not a real
Claude conversation resuming.

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


@contextmanager
def _launch_terminal(
    tmp_path, monkeypatch, *, terminal_user: str = "", landing_url: str = ""
) -> Iterator[str]:
    """Launch the web terminal with ``OSPREY_TERMINAL_{USER,LANDING_URL}`` set.

    The env vars are read by ``create_app`` (app.py:443-444) into ``app.state``,
    so they MUST be in ``os.environ`` before the factory runs. Companion backends
    (artifact server, panels) are bypassed via the same patch set the other
    web-terminal browser suites use.
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
def test_logout_and_return_resumes_same_session(tmp_path, monkeypatch, chromium_browser):
    """Full multi-user loop: header + logout render, logout navigates, return resumes.

    Asserts the client-side reconnect contract end to end: a stored PTY session id
    survives the logout->landing->return round trip unchanged, and on return the
    client reconnects to that SAME id via ``mode=resume`` rather than minting a
    fresh session. (See the module docstring on why the session id is seeded
    rather than produced by a live Claude turn.)
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

        # --- Represent an established PTY session (see module docstring) ---
        page.evaluate("(id) => localStorage.setItem('osprey-pty-session', id)", stored_id)
        captured = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
        assert captured == stored_id

        # Dismiss the first-visit welcome overlay so it can't intercept the click
        # (it covers the header on a fresh server session).
        page.evaluate(
            "() => { const o = document.getElementById('welcome-overlay'); if (o) o.remove(); }"
        )

        # --- Logout navigates the page to the landing origin ---
        page.click("#logout-btn")
        page.wait_for_url(lambda u: u.startswith(landing_url))
        assert page.url.startswith(landing_url)
        expect(page.locator("#landing-marker")).to_be_visible()

        # --- Return to the terminal origin: same stored id, mode=resume WS ---
        with page.expect_websocket() as return_ws:
            page.goto(base_url, wait_until="load")
        resume_url = return_ws.value.url
        assert f"session_id={stored_id}" in resume_url, (
            f"return visit should resume the stored session, got {resume_url}"
        )
        assert "mode=resume" in resume_url

        # localStorage id must survive the round trip unchanged (no fresh session).
        assert page.evaluate("() => localStorage.getItem('osprey-pty-session')") == stored_id

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
