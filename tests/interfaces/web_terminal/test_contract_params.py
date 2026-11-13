"""Browser tests for the shell<->panel micro-frontend contract (Phase 1).

Proves the three transports a host page (the web-terminal hub, or a
standalone panel opened directly) and an embedded panel use to talk to each
other, each of which is invisible to a FastAPI ``TestClient`` because it
requires a real browser evaluating real page script:

- **query = creation-time config.** ``?embedded=true`` -> ``applyEmbedded()``
  (design-system ``frame-params.js``) adds the ``embedded`` class to
  ``document.body``; ``?theme=<id>`` -> ``theme-boot.js`` applies
  ``data-theme`` before first paint and ``theme-manager.js`` follows it.
  See :func:`test_embedded_activity_page_hides_its_name_and_has_no_theme_control`.
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
- ``web_terminal`` ``session.html`` (the Activity page) session-change receiver --
  :func:`test_postmessage_session_change_session_html_rejects_foreign_origin`.
  ``/static/session.html`` is independently drivable: web_terminal's
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

import json
import time

import pytest

from tests.interfaces._launchers import (
    launch_artifacts,
    launch_channel_finder,
    launch_okf_panel,
    launch_web_terminal,
)
from tests.interfaces.web_terminal._surface import arm_terminal_tab

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# (1) hash = panel-owned deep-link
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
    with launch_okf_panel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}#control-system/channel-finding", wait_until="load")

        # Assert -- renderConcept() (app.js) set the reading-pane heading to
        # the concept's frontmatter title.
        expect(page.locator("h1.concept-title")).to_have_text("Channel Finding", timeout=10_000)
        # Assert -- highlightActive() marked the matching sidebar entry active.
        expect(
            page.locator(
                '.concept-link[aria-current="page"][data-concept-id="control-system/channel-finding"]'
            )
        ).to_be_attached(timeout=10_000)

        page.close()


# ---------------------------------------------------------------------------
# (2) postMessage = live push
# ---------------------------------------------------------------------------


def test_postmessage_theme_change_same_origin_only(tmp_path, monkeypatch, chromium_browser):
    """theme-manager's ``_handleMessage`` applies same-origin broadcasts, drops foreign ones.

    A genuine same-page ``postMessage`` always stamps ``event.origin`` as the
    page's own origin -- there is no way to make a real postMessage arrive
    with a spoofed origin -- so the acceptance step below is a real exercise
    of the accept path. The rejection step fakes a foreign origin via an
    in-page synthetic ``dispatchEvent(new MessageEvent(...))``.

    Loaded ``?embedded=true``, as the hub loads it: only a follower applies
    broadcasts at all. A standalone page runs theme-manager.js in the hub
    role (it owns its own display menu) and ignores them by design.
    """
    # Arrange
    with launch_channel_finder(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(f"{base_url}?theme=dark&embedded=true", wait_until="load")
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

    ``fetchArtifacts()`` posts ``{"key": <currentSessionId>, ...}`` to
    ``/api/artifacts/session-list`` whenever ``currentSessionId`` is set (and
    ``showAllSessions`` is false), and reads the unscoped ``GET /api/artifacts``
    otherwise, so the request (its URL and body) is the observable signal for
    whether the receiver actually updated ``currentSessionId``. The genuine handler also
    calls ``fetchArtifacts()`` itself on acceptance, so the positive-control
    request below needs no extra UI action to trigger it.
    """
    # Arrange
    with launch_artifacts(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        with page.expect_request(lambda r: "/api/artifacts" in r.url) as initial_info:
            page.goto(base_url, wait_until="load")
        assert "session_id" not in initial_info.value.url

        # The foreign message is dispatched synchronously, so were it accepted, its
        # fetch would be the first request recorded after this point.
        after_initial: list = []
        page.on(
            "request",
            lambda r: after_initial.append(r) if "/api/artifacts" in r.url else None,
        )

        # Act -- synthetic foreign-origin session-change.
        page.evaluate(
            """
            () => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-session-change', session_id: 'evil-session-999'},
            }))
            """
        )

        # Act -- genuine same-origin session-change (positive control).
        with page.expect_request(lambda r: "/api/artifacts" in r.url) as accepted_info:
            page.evaluate(
                "window.postMessage({type: 'osprey-session-change',"
                " session_id: 'contract-test-session'}, window.location.origin)"
            )
        # Assert -- accepted: the handler's own fetchArtifacts() call carries
        # the new session id, proving the rejection assertion above would
        # have caught a real leak.
        accepted = accepted_info.value
        assert accepted.url.endswith("/api/artifacts/session-list")
        assert accepted.method == "POST"
        assert json.loads(accepted.post_data)["key"] == "contract-test-session"

        # Assert -- rejected: the foreign message caused no request of its own.
        assert after_initial[0] == accepted
        for request in after_initial:
            assert "evil-session-999" not in request.url
            assert "evil-session-999" not in (request.post_data or "")

        page.close()


def test_postmessage_session_change_session_html_rejects_foreign_origin(
    tmp_path, monkeypatch, chromium_browser
):
    """web_terminal's /static/session.html session-change receiver rejects foreign origin.

    The Activity page reads ``/api/session-activity`` for the session it
    shows, and names a session key as ``?session_id=<key>`` on that request,
    so the request URL is the observable signal for whether the receiver moved
    the page. Both messages carry a well-formed session key: the receiver
    drops anything that is not one, so a malformed key would be rejected by
    that check and prove nothing about the origin guard.
    """
    foreign = "99999999-8888-4777-8666-555555555555"
    accepted = "11111111-2222-4333-8444-555555555555"

    def _is_read(request) -> bool:
        return "/api/session-activity" in request.url and "/result" not in request.url

    # Arrange
    with launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        reads: list[str] = []
        page.on("request", lambda r: reads.append(r.url) if _is_read(r) else None)

        with page.expect_request(_is_read) as initial_info:
            page.goto(f"{base_url}/static/session.html", wait_until="load")
        assert "session_id" not in initial_info.value.url

        # Act -- synthetic foreign-origin session-change.
        page.evaluate(
            """
            key => window.dispatchEvent(new MessageEvent('message', {
                origin: 'https://evil.example',
                data: {type: 'osprey-session-change', session_id: key},
            }))
            """,
            foreign,
        )
        # Assert -- rejected: the page did not move, so the read its Refresh
        # button sends still names no session, and no read ever named the
        # foreign one.
        with page.expect_request(_is_read) as rejected_info:
            page.locator("#refresh").click()
        assert "session_id" not in rejected_info.value.url
        assert not any(foreign in url for url in reads), reads

        # Act -- genuine same-origin session-change (positive control). The
        # receiver moves the page, which reads the named session at once.
        with page.expect_request(_is_read) as accepted_info:
            page.evaluate(
                "key => window.postMessage({type: 'osprey-session-change',"
                " session_id: key}, window.location.origin)",
                accepted,
            )
        # Assert -- accepted, proving the rejection assertion above would
        # have caught a real leak.
        assert f"session_id={accepted}" in accepted_info.value.url

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
    with launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        arm_terminal_tab(page)
        sent_frames: list[str] = []

        def _on_websocket(ws) -> None:
            ws.on("framesent", lambda payload: sent_frames.append(payload))

        page.on("websocket", _on_websocket)

        page.goto(base_url, wait_until="load")
        expect(
            page.locator('button.panel-rail-button[data-panel-id="terminal"]:not(.disabled)')
        ).to_be_visible(timeout=10_000)
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


# ---------------------------------------------------------------------------
# (3) chrome contract: query config, embedded-hide, D15 reload-strip
# ---------------------------------------------------------------------------
# The Activity page (session.html) is the one web-terminal page that opens in
# a tab of its own as well as framed in the hub. It is a static page under the
# hub's /static mount, not a second app, so `launch_web_terminal` serves it.
# It carries no theme control: it shows the theme the terminal is in, and the
# theme is picked in the hub's More menu. So the embedded-hide rule is asserted
# on the page's own name, and the D15 reload-strip on the hub, whose appearance
# rows are the web terminal's theme control.

_ACTIVITY_PAGE = "/static/session.html"


def test_embedded_activity_page_hides_its_name_and_has_no_theme_control(
    tmp_path, monkeypatch, chromium_browser
):
    """``?embedded=true&theme=dark`` on the Activity page: framed, dark, unnamed.

    The query=config arm: ``applyEmbedded()`` adds ``body.embedded`` and the
    creation-time ``?theme=`` applies. Framed, the hub's own tab names the
    page, so the page's ``body.embedded .act-page-name { display: none }``
    rule hides its name; in a tab of its own the name shows. Neither way does
    the page carry a theme control.
    """
    # Arrange
    with launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}{_ACTIVITY_PAGE}?embedded=true&theme=dark", wait_until="load")

        # Assert -- applyEmbedded() ran.
        expect(page.locator("body.embedded")).to_have_count(1)
        # Assert -- the creation-time ?theme= applies in the embedded role too
        # (pre-paint theme-boot.js + the theme-manager follower). 'dark' rather
        # than the auto default, which is 'light' under headless Chromium, so
        # the query param is what drove it.
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)
        # Assert -- the page's own name is hidden, and it has no theme control.
        expect(page.locator(".act-page-name")).to_have_count(1)
        expect(page.locator(".act-page-name")).to_be_hidden()
        expect(page.locator("osprey-theme-switcher")).to_have_count(0)

        # Act -- the same page in a tab of its own.
        page.goto(f"{base_url}{_ACTIVITY_PAGE}", wait_until="load")

        # Assert -- not embedded: the name shows; still no theme control.
        expect(page.locator("body.embedded")).to_have_count(0)
        expect(page.locator(".act-page-name")).to_be_visible()
        expect(page.locator("osprey-theme-switcher")).to_have_count(0)

        page.close()


def test_theme_toggle_strips_stale_query_param_and_survives_reload(
    tmp_path, monkeypatch, chromium_browser
):
    """In the hub, picking Light over a ``?theme=dark`` URL sticks across a reload.

    D15: the pick strips ``?theme=`` from the URL, so the reload resolves the
    stored pick instead of resurrecting the param. Starts from a real
    ``?theme=dark`` query param (not merely its absence) so the post-pick
    assertion proves setTheme()'s ``history.replaceState`` strip actually
    removed something, rather than passing vacuously on a URL that never had
    the param to begin with.
    """
    # Arrange
    with launch_web_terminal(tmp_path, monkeypatch) as base_url:
        # A dark OS preference, so the auto-resolved theme a reload would fall
        # back to without a stored pick is dark — the opposite of the pick.
        page = chromium_browser.new_page(color_scheme="dark")
        page.goto(f"{base_url}/?theme=dark", wait_until="load")
        # Assert -- the query param applied.
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- pick Light in the More menu's appearance rows.
        page.click("#more-btn")
        expect(page.locator("#more-menu")).to_be_visible()
        page.click('#more-menu [role="menuitemradio"][data-appearance="light"]')

        # Assert -- the theme is light, and the leftover ?theme=dark is gone
        # from the URL immediately.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)
        assert "theme=" not in page.url

        # Act -- reload.
        page.reload(wait_until="load")

        # Assert -- the stale param can't be resurrected because it was
        # actually stripped (not just visually ignored): reload falls back to
        # localStorage, then the OS preference. The hub persisted the pick, so
        # it comes back light rather than the OS's dark.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)
        assert "theme=" not in page.url

        page.close()


def test_channel_finder_embedded_non_occlusion(tmp_path, monkeypatch, chromium_browser):
    """channel_finder embedded opens on its panel toolbar, with no dead band.

    The panel toolbar is the same embedded and standalone (DESIGN.md §12.3),
    so it is the first thing in the frame, right under the hub's tile bar.
    The view scrolls between it and the corpus strip, and the strip carries
    the pipeline switcher: it must stay rendered inside the viewport, below
    the view rather than over it, for the panel to remain switchable when
    embedded.
    """
    # Arrange
    with launch_channel_finder(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # Assert -- the panel toolbar is at the very top of the frame.
        toolbar = page.locator("#cf-toolbar")
        expect(toolbar).to_be_visible(timeout=10_000)
        toolbar_box = toolbar.bounding_box()
        assert toolbar_box is not None
        assert toolbar_box["y"] == 0

        # Assert -- the pipeline switcher is rendered and positioned inside the viewport.
        # ``load`` fires before the switcher's layout settles, so on a loaded runner
        # bounding_box() can catch it at width/height 0; wait for it to be visible
        # (Playwright's visibility check requires a non-empty box) before measuring.
        switcher = page.locator("#pipeline-switcher")
        expect(switcher).to_be_visible(timeout=10_000)
        box = switcher.bounding_box()
        assert box is not None, "#pipeline-switcher has no bounding box -- is it rendered?"
        viewport = page.viewport_size
        assert viewport is not None
        assert box["width"] > 0 and box["height"] > 0
        assert 0 <= box["y"] <= viewport["height"]

        # Assert -- the view ends where the strip begins instead of running under
        # it, and the strip is the last thing in the frame.
        main_box = page.locator("#app-main").bounding_box()
        strip_box = page.locator("#corpus-strip").bounding_box()
        assert main_box is not None and strip_box is not None
        assert main_box["y"] + main_box["height"] == pytest.approx(strip_box["y"], abs=1)
        assert strip_box["y"] + strip_box["height"] == pytest.approx(viewport["height"], abs=1)

        page.close()
