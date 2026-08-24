"""Browser tests: the per-session posture toggle, end to end.

The posture badge is the operator's only route between ``writes`` and
``sandbox`` on a live session, and every interesting part of it is client-side
behavior a FastAPI TestClient cannot observe: the badge only exists once
``terminal.js`` has reported a session id, the confirm dialog is built in the
DOM by ``posture-badge.js``, and the badge repaints from a *re-read* of
``GET /api/terminal/posture`` rather than from what the module last POSTed. A
real browser is what proves the badge an operator looks at agrees with the
store the executor will spawn under.

Coverage (one test each):

  (a) confirming the sandbox direction moves the badge to
      ``data-posture="sandbox"`` — and the server's own store agrees, so the
      badge is not merely painting its own optimism.
  (b) the confirm dialog appears in BOTH directions, naming the direction it
      is about, and Cancel leaves the posture exactly as it was (checked on
      the badge *and* on the server) — a toggle that fired on the way to the
      dialog would be the worst possible failure of a confirm step.
  (c) toggling a session the server has never seen surfaces the route's own
      409 wording ("send one prompt first") in the dialog's error banner,
      with the dialog kept up to carry it.

Session bootstrapping. The badge needs a session id, and one arrives the way
it does in production: a plain page load opens a NEW terminal WebSocket, and
the route mints the session UUID itself (it dictates it on the CLI's command
line) and confirms it immediately in a ``session_info`` frame — no Claude
binary and no session discovery involved. The id the card settled on is then
read back from ``localStorage['osprey-pty-session']``, which ``terminal.js``
writes on that same frame. The PTY command is a long-lived ``sleep`` because
the route appends ``--session-id``/``--resume`` arguments that ``echo`` would
choke on, and a PTY that exits during the resume-failover window would make
the client drop the very id under test.

``SessionDiscovery.snapshot_session_ids`` is patched to a set the test owns
(the same seam ``test_posture_routes.py`` uses): "this session exists on disk"
is otherwise only true after a real model turn has written a ``.jsonl``, and
it is exactly the distinction case (c) turns on. The posture store is pinned
to a tmp shared-data root so no test writes into the real agent-data tree.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_posture_toggle_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import requests
import yaml

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# Every wait in this file uses one generous bound rather than a tuned-per-step
# one. These suites run under parallel load where a step that normally takes
# milliseconds can take seconds, and a tight bound buys nothing: a state that
# is never going to arrive fails the assertion either way, only later.
TIMEOUT = 15_000

BADGE = ".posture-badge"
# `:not([data-closing])` is the contract for "the dialog is OPEN". Dismissal is
# marked by the attribute and the node is only detached after the fade (~300ms),
# so asserting on detachment would be asserting on an animation.
OPEN_MODAL = ".posture-modal-overlay:not([data-closing])"
MODAL_TITLE = f"{OPEN_MODAL} .posture-modal-title"
MODAL_CONFIRM = f"{OPEN_MODAL} .posture-modal-confirm"
MODAL_CANCEL = f"{OPEN_MODAL} .posture-modal-cancel"
MODAL_ERROR = f"{OPEN_MODAL} .posture-modal-error"

# A PTY command that outlives the test AND tolerates the arguments the
# websocket route appends (``--session-id <uuid>`` on a new session,
# ``--resume <uuid>`` on the reconnect the toggle performs). ``echo``/``sleep``
# would exit or error on those, and an exit inside the resume-failover window
# makes terminal.js discard the session id the badge is pointed at.
_LONG_LIVED_SHELL = [sys.executable, "-c", "import time; time.sleep(3600)"]


# ---------------------------------------------------------------------------
# Live-server helpers
# ---------------------------------------------------------------------------


@contextmanager
def _posture_hub(
    tmp_path: Path,
    *,
    known_ids: set[str],
    writes_enabled: bool = True,
) -> Iterator[str]:
    """Launch a real web-terminal hub wired for the posture toggle.

    The companion-backend patches are the ones every hub browser suite uses.
    Three more are specific to this feature:

    * ``resolve_shared_data_root`` — the posture store is written through to
      disk on every POST; pinning the root to ``tmp_path`` keeps that off the
      real agent-data tree.
    * ``snapshot_session_ids`` — POST refuses (409) an id that names no session
      file, and no session file is ever written here. *known_ids* is the test's
      own set and is read on every call, so a test can add the id the server
      minted once the page has told it what that id is.
    * ``config_path`` — ``control_system.writes_enabled`` is what the GET
      reports as ``rendered_writes_enabled`` and what gates the ``writes``
      direction. A hub with no config is a writes-OFF render, where the badge
      is deliberately unclickable in one direction.

    Yields:
        The hub's base URL.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir(exist_ok=True)
    shared_root = tmp_path / "shared_agent_data"
    shared_root.mkdir(exist_ok=True)
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump({"control_system": {"writes_enabled": writes_enabled}}),
        encoding="utf-8",
    )

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
        patch(
            "osprey_connectors.workspace.resolve_shared_data_root",
            return_value=shared_root,
        ),
        patch(
            "osprey.interfaces.web_terminal.session_discovery.SessionDiscovery"
            ".snapshot_session_ids",
            side_effect=lambda *_args, **_kwargs: set(known_ids),
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=list(_LONG_LIVED_SHELL), config_path=config)
        with _run_app_server(app) as base_url:
            yield base_url


def _open_hub_with_session(browser: Browser, base_url: str) -> tuple[Page, str]:
    """Open the hub and wait until the badge is painted for a live session.

    A visible badge already means the whole chain ran: the terminal connected,
    the route confirmed a session id, and ``GET /api/terminal/posture``
    answered for it (the badge stays hidden until a read succeeds).

    Returns:
        (page, session_id) — the session the card settled on, read back from
        the same storage key ``terminal.js`` writes on the ``session_info``
        frame.
    """
    page = browser.new_page()
    page.goto(base_url, wait_until="domcontentloaded")

    badge = page.locator(BADGE)
    expect(badge).to_be_visible(timeout=TIMEOUT)
    # A fresh session has no store entry, and "no entry" is the render's
    # baseline — reported as ``writes``.
    expect(badge).to_have_attribute("data-posture", "writes", timeout=TIMEOUT)

    session_id = page.evaluate("() => localStorage.getItem('osprey-pty-session')")
    assert session_id, "the terminal card never settled on a session id"
    return page, session_id


def _server_posture(base_url: str, session_id: str) -> dict:
    """What the server itself says this session's posture is.

    The badge is *supposed* to mirror this (it re-reads after every mutation),
    so asserting both is what distinguishes a real toggle from a badge that
    repainted from its own optimism — and, on the Cancel paths, proves nothing
    was written rather than merely nothing being shown.
    """
    resp = requests.get(
        f"{base_url}/api/terminal/posture",
        params={"session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_confirming_sandbox_moves_badge_and_store(tmp_path, chromium_browser):
    """Toggling to sandbox repaints the badge AND lands in the server's store.

    The full operator path: click the badge, confirm the dialog, and the card
    comes back sandboxed. The store assertion is the load-bearing half — it is
    the value ``_build_extra_env`` reads on the respawn, so a badge that agreed
    with nothing would be a session the operator believes is sandboxed and the
    executor does not.
    """
    known_ids: set[str] = set()

    with _posture_hub(tmp_path, known_ids=known_ids) as base_url:
        page, session_id = _open_hub_with_session(chromium_browser, base_url)
        try:
            # The session the route just minted counts as started on disk: the
            # posture is being set on a session the operator has been talking
            # to, which is the only case POST accepts.
            known_ids.add(session_id)

            page.locator(BADGE).click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            expect(page.locator(BADGE)).to_have_attribute(
                "data-posture", "sandbox", timeout=TIMEOUT
            )
            # Success dismisses the dialog; only a refusal keeps it up.
            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)

            assert _server_posture(base_url, session_id)["posture"] == "sandbox"
        finally:
            page.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_both_directions_confirm_and_cancel_changes_nothing(tmp_path, chromium_browser):
    """Every toggle asks first — in both directions — and Cancel is a true no-op.

    The dialog is unconditional by design (no remembered acknowledgment), so
    both directions are exercised on one live session: writes -> sandbox is
    cancelled, then performed, then sandbox -> writes is cancelled. After each
    cancellation the badge and the server store must both still read what they
    read before the badge was clicked.
    """
    known_ids: set[str] = set()

    with _posture_hub(tmp_path, known_ids=known_ids, writes_enabled=True) as base_url:
        page, session_id = _open_hub_with_session(chromium_browser, base_url)
        try:
            known_ids.add(session_id)

            # --- Direction 1 (writes -> sandbox), cancelled ---
            page.locator(BADGE).click()
            expect(page.locator(MODAL_TITLE)).to_have_text("Sandbox this session?", timeout=TIMEOUT)
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(BADGE)).to_have_attribute("data-posture", "writes", timeout=TIMEOUT)
            assert _server_posture(base_url, session_id)["posture"] == "writes"

            # --- Direction 1, confirmed: now the session is sandboxed ---
            page.locator(BADGE).click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()
            expect(page.locator(BADGE)).to_have_attribute(
                "data-posture", "sandbox", timeout=TIMEOUT
            )

            # --- Direction 2 (sandbox -> writes), cancelled ---
            # Available at all only because this render has writes enabled; the
            # badge would be disabled otherwise.
            page.locator(BADGE).click()
            expect(page.locator(MODAL_TITLE)).to_have_text(
                "Allow writes for this session?", timeout=TIMEOUT
            )
            page.locator(MODAL_CANCEL).click()

            expect(page.locator(OPEN_MODAL)).to_have_count(0, timeout=TIMEOUT)
            expect(page.locator(BADGE)).to_have_attribute(
                "data-posture", "sandbox", timeout=TIMEOUT
            )
            assert _server_posture(base_url, session_id)["posture"] == "sandbox"
        finally:
            page.close()


@pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="playwright not installed")
def test_unstarted_session_surfaces_the_409_remedy(tmp_path, chromium_browser):
    """A session with no file on disk is refused, and the dialog says why.

    ``known_ids`` stays empty, so the id the card is on names no session file —
    the state a terminal is in before its first prompt. The route answers 409
    with a dict detail, and the badge module unwraps ``detail.message`` (rather
    than stringifying the dict to "[object Object]") into the dialog's error
    banner. The remedy sentence is the whole point of the refusal, so the
    dialog stays up to carry it and nothing is written.
    """
    with _posture_hub(tmp_path, known_ids=set()) as base_url:
        page, session_id = _open_hub_with_session(chromium_browser, base_url)
        try:
            page.locator(BADGE).click()
            expect(page.locator(MODAL_CONFIRM)).to_be_visible(timeout=TIMEOUT)
            page.locator(MODAL_CONFIRM).click()

            error = page.locator(MODAL_ERROR)
            expect(error).to_be_visible(timeout=TIMEOUT)
            expect(error).to_contain_text("send one prompt first", timeout=TIMEOUT)

            # The dialog is still the open one — a refusal must not dismiss it,
            # or the operator never reads the sentence that tells them what to
            # do next.
            expect(page.locator(OPEN_MODAL)).to_have_count(1, timeout=TIMEOUT)
            # Nothing was terminated and nothing was stored.
            expect(page.locator(BADGE)).to_have_attribute("data-posture", "writes", timeout=TIMEOUT)
            assert _server_posture(base_url, session_id)["posture"] == "writes"
        finally:
            page.close()
