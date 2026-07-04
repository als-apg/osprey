"""Browser tests for the shell<->panel micro-frontend contract (Phase 1).

Proves the three transports a host page (the web-terminal hub, or a
standalone panel opened directly) and an embedded panel use to talk to each
other, each of which is invisible to a FastAPI ``TestClient`` because it
requires a real browser evaluating real page script:

- **query = creation-time config.** ``?embedded=true`` -> ``applyEmbedded()``
  (design-system ``frame-params.js``) adds the ``embedded`` class to
  ``document.body``; ``?theme=<id>`` -> ``theme-boot.js`` applies
  ``data-theme`` before first paint and ``theme-manager.js`` follows it.
  See :func:`test_query_params_configure_embedded_and_theme`.
- **hash = panel-owned deep-link.** okf_panel reads its own
  ``location.hash`` and routes to that concept on load
  (``readPanelParams``/``bootFromParams`` in okf_panel's ``app.js``); the hub
  sets no hash at all. See :func:`test_hash_deep_links_to_concept`.
- **postMessage = live push.** Senders post with target origin
  ``window.location.origin``; receivers guard with
  ``if (e.origin !== window.location.origin) return;``. A real same-page
  ``postMessage`` always stamps ``event.origin`` as the page's own origin, so
  it can only exercise the *accept* path; the *reject* path is exercised with
  an in-page synthetic ``window.dispatchEvent(new MessageEvent('message',
  {origin: 'https://evil.example', ...}))``, which is the only way to give
  ``event.origin`` a value a real cross-document postMessage could never
  produce here.

All four postMessage receivers are driven directly in a live browser (no
review-based fallback was needed):

- ``theme-manager.js`` ``_handleMessage`` -- :func:`test_postmessage_theme_change_same_origin_only`
- ``artifacts`` gallery.js session-change receiver -- :func:`test_postmessage_session_change_gallery_rejects_foreign_origin`
- ``web_terminal`` ``session.html`` session-change receiver --
  :func:`test_postmessage_session_change_session_html_rejects_foreign_origin`.
  ``/static/session.html`` turned out to be independently drivable: web_terminal's
  ``configure_interface_app(app, static_dir=STATIC_DIR)`` mounts the interface's
  whole ``static/`` directory at ``/static``, and ``session.html`` lives directly
  under it, so ``GET /static/session.html`` serves it like any other static asset.
- ``web_terminal`` app.js paste-to-terminal receiver -- :func:`test_postmessage_paste_to_terminal_rejects_foreign_origin`

Each postMessage test also includes a same-origin "positive control" after
the foreign-origin rejection check: a genuine same-origin message that IS
expected to take effect. This guards against a vacuous pass -- proving the
detection technique used for the rejection assertion (a request URL, or a
WebSocket frame) is actually capable of observing the effect it claims is
absent.

Run:
    .venv/bin/python -m pytest tests/interfaces/web_terminal/test_contract_params.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import time

import pytest

from tests.interfaces.test_load_smokes import (
    _launch_ariel,
    _launch_artifacts,
    _launch_channel_finder,
    _launch_lattice_dashboard,
    _launch_okf_panel,
    _launch_tuning,
    _launch_web_terminal,
)

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# (1) query = creation-time config: ?embedded=true, ?theme=<id>
# ---------------------------------------------------------------------------


# Every panel that the hub embeds shares one reader (frame-params.applyEmbedded())
# and one theme follower (theme-manager + pre-paint theme-boot.js), but each calls
# applyEmbedded() at a different site/timing (inline top-level, inside init(),
# inside checkEmbedded(), inside a DOMContentLoaded handler) -- so the query=config
# arm is asserted for ALL five, not just one, to prove no per-panel migration
# regressed the observable outcome.
_EMBEDDED_PANELS = [
    ("channel_finder", _launch_channel_finder),
    ("okf_panel", _launch_okf_panel),
    ("ariel", _launch_ariel),
    ("tuning", _launch_tuning),
    ("lattice_dashboard", _launch_lattice_dashboard),
]


@pytest.mark.parametrize(
    ("panel_name", "launch"),
    _EMBEDDED_PANELS,
    ids=[name for name, _ in _EMBEDDED_PANELS],
)
def test_query_params_configure_embedded_and_theme(
    panel_name, launch, tmp_path, monkeypatch, chromium_browser
):
    """``?embedded=true`` adds ``body.embedded``; ``?theme=dark`` applies data-theme -- every panel.

    Drives each of the five embedded panels with the same creation-time query
    config and asserts the observable outcome -- the ``embedded`` body class
    (from ``frame-params.applyEmbedded()``) and the requested ``data-theme``
    (from pre-paint ``theme-boot.js`` + the ``theme-manager`` follower).
    Requesting theme id 'dark' -- rather than relying on the auto-resolved
    default, which is 'light' under headless Chromium's default no-preference
    color scheme -- proves the query param, not the auto default, drove the
    result.
    """
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}?embedded=true&theme=dark", wait_until="load")

        # Assert -- applyEmbedded() (frame-params.js) added the embedded class.
        expect(page.locator("body.embedded")).to_have_count(1)
        # Assert -- theme-boot.js (pre-paint) + theme-manager.js applied 'dark'.
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        page.close()


# ---------------------------------------------------------------------------
# (2) hash = panel-owned deep-link
# ---------------------------------------------------------------------------


def test_hash_deep_links_to_concept(tmp_path, monkeypatch, chromium_browser):
    """okf_panel's own ``#<conceptId>`` hash routes to that concept on load.

    The hub sets no hash at all (the dead ``#/sessions?project=`` grammar was
    removed) -- this transport is entirely panel-owned. Drives okf_panel
    standalone with a hash matching a real concept in the shared fixture
    bundle (``tests/interfaces/okf_panel/fixtures/bundle/control-system/
    channel-finding.md``, frontmatter title "Channel Finding") and asserts
    the reading pane actually rendered that concept -- not the default
    structure overview ``bootFromParams()`` falls back to when there is no
    hash.
    """
    # Arrange
    with _launch_okf_panel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}#control-system/channel-finding", wait_until="load")

        # Assert -- renderConcept() (app.js) set the reading-pane heading to
        # the concept's frontmatter title.
        expect(page.locator("h1.concept-title")).to_have_text("Channel Finding", timeout=10_000)
        # Assert -- highlightActive() marked the matching sidebar entry active.
        expect(
            page.locator('.concept-link.active[data-concept-id="control-system/channel-finding"]')
        ).to_be_attached(timeout=10_000)

        page.close()


# ---------------------------------------------------------------------------
# (3) postMessage = live push
# ---------------------------------------------------------------------------


def test_postmessage_theme_change_same_origin_only(tmp_path, monkeypatch, chromium_browser):
    """theme-manager's ``_handleMessage`` applies same-origin broadcasts, drops foreign ones.

    A genuine same-page ``postMessage`` always stamps ``event.origin`` as the
    page's own origin -- there is no way to make a real postMessage arrive
    with a spoofed origin -- so the acceptance step below is a real exercise
    of the accept path. The rejection step fakes a foreign origin via an
    in-page synthetic ``dispatchEvent(new MessageEvent(...))``.
    """
    # Arrange
    with _launch_channel_finder(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(f"{base_url}?theme=dark", wait_until="load")
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- genuine same-origin postMessage.
        page.evaluate(
            "window.postMessage({type: 'osprey-theme-change', theme: 'light'},"
            " window.location.origin)"
        )
        # Assert -- accepted: theme changed.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)

        # Act -- synthetic foreign-origin MessageEvent.
        page.evaluate(
            """
            () => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-theme-change', theme: 'dark'},
            }))
            """
        )
        # Assert -- rejected: theme-manager.js's origin guard
        # (`if (event.origin !== window.location.origin) return;`) dropped it
        # before touching data-theme, so it is still 'light' from the
        # accepted message above, not 'dark' from the rejected one.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)

        page.close()


def test_postmessage_session_change_gallery_rejects_foreign_origin(
    tmp_path, monkeypatch, chromium_browser
):
    """artifacts gallery.js's session-change receiver drops foreign-origin messages.

    ``fetchArtifacts()`` appends ``?session_id=<currentSessionId>`` to its own
    request whenever ``currentSessionId`` is set (and ``showAllSessions`` is
    false), so the request URL is the observable signal for whether the
    receiver actually updated ``currentSessionId``. The genuine handler also
    calls ``fetchArtifacts()`` itself on acceptance, so the positive-control
    request below needs no extra UI action to trigger it.
    """
    # Arrange
    with _launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        with page.expect_request(lambda r: "/api/artifacts" in r.url) as initial_info:
            page.goto(base_url, wait_until="load")
        assert "session_id" not in initial_info.value.url

        # Act -- synthetic foreign-origin session-change.
        page.evaluate(
            """
            () => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-session-change', session_id: 'evil-session-999'},
            }))
            """
        )
        # Assert -- rejected: currentSessionId did not change, so an
        # independent refresh click still fetches with no session_id.
        with page.expect_request(lambda r: "/api/artifacts" in r.url) as rejected_info:
            page.locator("#refresh-btn").click()
        assert "evil-session-999" not in rejected_info.value.url
        assert "session_id" not in rejected_info.value.url

        # Act -- genuine same-origin session-change (positive control).
        with page.expect_request(lambda r: "/api/artifacts" in r.url) as accepted_info:
            page.evaluate(
                "window.postMessage({type: 'osprey-session-change',"
                " session_id: 'contract-test-session'}, window.location.origin)"
            )
        # Assert -- accepted: the handler's own fetchArtifacts() call carries
        # the new session id, proving the rejection assertion above would
        # have caught a real leak.
        assert "session_id=contract-test-session" in accepted_info.value.url

        page.close()


def test_postmessage_session_change_session_html_rejects_foreign_origin(
    tmp_path, monkeypatch, chromium_browser
):
    """web_terminal's /static/session.html session-change receiver rejects foreign origin.

    ``session.html`` is served directly by web_terminal's own ``/static``
    mount (``configure_interface_app(app, static_dir=STATIC_DIR)`` in
    ``osprey/interfaces/web_terminal/app.py`` mounts the whole static
    directory, and ``session.html`` lives at its top level), so it is
    independently drivable rather than needing a review-based fallback.
    ``apiFetch()`` appends ``&session_id=<currentSessionId>`` to every
    request once ``currentSessionId`` is set, so the request URL for a view
    fetch is the observable signal for whether the receiver updated it.
    """
    # Arrange
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        with page.expect_request(lambda r: "/api/session-agents" in r.url) as initial_info:
            page.goto(f"{base_url}/static/session.html", wait_until="load")
        assert "session_id" not in initial_info.value.url

        # Act -- synthetic foreign-origin session-change.
        page.evaluate(
            """
            () => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-session-change', session_id: 'evil-session-999'},
            }))
            """
        )
        # Assert -- rejected: currentSessionId did not change; switching to
        # the Tool Log view still fetches with no (foreign) session_id.
        with page.expect_request(lambda r: "/api/session-log" in r.url) as rejected_info:
            page.locator('.pill[data-view="toollog"]').click()
        assert "evil-session-999" not in rejected_info.value.url
        assert "session_id" not in rejected_info.value.url

        # Act -- genuine same-origin session-change (positive control). The
        # handler calls refreshActive() itself, re-fetching the currently
        # active view (Tool Log, from the click above) with the new session id.
        with page.expect_request(lambda r: "/api/session-log" in r.url) as accepted_info:
            page.evaluate(
                "window.postMessage({type: 'osprey-session-change',"
                " session_id: 'contract-test-session'}, window.location.origin)"
            )
        # Assert -- accepted, proving the rejection assertion above would
        # have caught a real leak.
        assert "session_id=contract-test-session" in accepted_info.value.url

        page.close()


def test_postmessage_paste_to_terminal_rejects_foreign_origin(
    tmp_path, monkeypatch, chromium_browser
):
    """web_terminal app.js's paste bridge drops foreign-origin messages.

    ``pasteToTerminal()`` (terminal.js) sends the pasted text straight over
    the PTY WebSocket, so a real network-level ``framesent`` event -- not a
    JS-level spy -- is the observable signal. The test waits for the
    ``{"type": "resize"}`` handshake frame terminal.js's ``onOpen`` sends
    immediately once connected before dispatching either message:
    ``pasteToTerminal()`` silently no-ops while the socket isn't OPEN yet
    (api.js's ``send()`` guards on ``ws.readyState === WebSocket.OPEN``), so
    dispatching before the socket opens would make the rejection assertion
    pass vacuously. (``#session-led.active`` was tried first but is not a
    reliable readiness signal here: this test's fixture shell command is
    ``echo hello``, which exits almost instantly, and the LED is removed as
    soon as the resulting ``{"type": "exit"}`` message arrives -- even though
    the socket itself stays open and paste still works.)
    """
    # Arrange
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        sent_frames: list[str] = []

        def _on_websocket(ws) -> None:
            ws.on("framesent", lambda payload: sent_frames.append(payload))

        page.on("websocket", _on_websocket)

        page.goto(base_url, wait_until="load")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_visible(timeout=10_000)
        # Confirm the socket is actually OPEN (not just created) by polling
        # the already-attached framesent listener for the resize handshake
        # frame terminal.js's onOpen sends first. A one-shot
        # ws.wait_for_event() would race: on this fixture's near-instant
        # 'echo hello' shell command, the resize frame is typically sent (and
        # the listener above already recorded it) well before this line runs,
        # so waiting for a *future* framesent event would time out on a frame
        # that already happened.
        deadline = time.monotonic() + 10.0
        while not any("resize" in frame for frame in sent_frames):
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"Never observed a 'resize' handshake frame on the terminal "
                    f"WebSocket within 10s; frames seen so far: {sent_frames}"
                )
            page.wait_for_timeout(50)

        # Act -- synthetic foreign-origin paste.
        page.evaluate(
            """
            () => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-paste-to-terminal', text: 'contract-test-foreign-paste'},
            }))
            """
        )
        # Best-effort settle: proving an ABSENCE of a network-level side
        # effect has no DOM/state to auto-wait on, unlike the other arms.
        page.wait_for_timeout(300)
        # Assert -- rejected: nothing sent over the PTY socket.
        assert not any("contract-test-foreign-paste" in frame for frame in sent_frames), sent_frames

        # Act -- genuine same-origin paste (positive control).
        page.evaluate(
            "window.postMessage({type: 'osprey-paste-to-terminal',"
            " text: 'contract-test-accepted-paste'}, window.location.origin)"
        )
        page.wait_for_timeout(300)
        # Assert -- accepted, proving the rejection assertion above would
        # have caught a real leak.
        assert any("contract-test-accepted-paste" in frame for frame in sent_frames), sent_frames

        page.close()
