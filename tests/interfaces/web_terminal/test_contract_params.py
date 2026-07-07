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


# ---------------------------------------------------------------------------
# (4) chrome contract: <osprey-theme-switcher> + embedded-hide + D15 reload-strip
# ---------------------------------------------------------------------------
# Every panel's theme toggle is the shared `<osprey-theme-switcher>`
# component, and `applyEmbedded()` is wired into each of the 6 panels
# below -- this is the automated proof that the chrome contract holds on
# every panel, not just the one or two it was developed against.
# web_terminal's own session.html joins as fleet page 7, via a
# path-override on `_launch_web_terminal` (same hub server
# test_load_smokes.py's own '/' case boots) rather than a distinct
# launcher -- session.html is a static page under the hub's /static
# mount, not a second app.
# `branding_selector` is the per-page standalone-only element the D15
# narrowing keeps hidden in embedded mode; okf_panel has no branding
# chrome of its own to hide, so its entry is `None` and the branding
# assertion is skipped for it.
_CHROME_CONTRACT_PANELS = [
    ("ariel", _launch_ariel, "", ".logo"),
    ("artifacts", _launch_artifacts, "", ".logo"),
    ("channel_finder", _launch_channel_finder, "", ".app-logo"),
    ("tuning", _launch_tuning, "", ".tuning-header"),
    ("lattice_dashboard", _launch_lattice_dashboard, "", ".topbar-logo"),
    ("okf_panel", _launch_okf_panel, "", None),
    ("web_terminal_session", _launch_web_terminal, "/static/session.html", "header h1"),
]


@pytest.mark.parametrize(
    ("panel_name", "launch", "path", "branding_selector"),
    _CHROME_CONTRACT_PANELS,
    ids=[name for name, _, _, _ in _CHROME_CONTRACT_PANELS],
)
def test_embedded_hides_branding_and_switcher(
    panel_name, launch, path, branding_selector, tmp_path, monkeypatch, chromium_browser
):
    """``?embedded=true`` hides the page's own branding AND the theme switcher.

    The switcher's own D15 rule (``body.embedded osprey-theme-switcher {
    display: none }``, injected once by osprey-theme-switcher.js itself) is
    what hides it -- no per-panel CSS is needed for that half of the
    contract. The branding selector, by contrast, is each page's own
    pre-existing ``body.embedded <selector> { display: none }`` rule; this
    proves the switcher rollout didn't disturb it.
    """
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}{path}?embedded=true", wait_until="load")

        # Assert -- applyEmbedded() ran.
        expect(page.locator("body.embedded")).to_have_count(1)
        # Assert -- the switcher is hidden fleet-wide by its own injected rule.
        assert (
            page.evaluate(
                "getComputedStyle(document.querySelector('osprey-theme-switcher')).display"
            )
            == "none"
        )
        # Assert -- the page's own branding is hidden (skipped where none exists).
        if branding_selector:
            assert (
                page.evaluate(
                    f"getComputedStyle(document.querySelector('{branding_selector}')).display"
                )
                == "none"
            )

        page.close()


@pytest.mark.parametrize(
    ("panel_name", "launch", "path", "branding_selector"),
    _CHROME_CONTRACT_PANELS,
    ids=[name for name, _, _, _ in _CHROME_CONTRACT_PANELS],
)
def test_switcher_present_and_toggles_theme_standalone(
    panel_name, launch, path, branding_selector, tmp_path, monkeypatch, chromium_browser
):
    """Standalone (no ``?embedded``), the switcher is visible and its click toggles the theme.

    Starts from an explicit ``?theme=dark`` (rather than relying on the
    auto-resolved default) so the post-click assertion -- 'light' -- proves
    the click actually drove ``toggleTheme()``, not a coincidental default.
    """
    del branding_selector  # unused here; shared parametrization with the embedded test above
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}{path}?theme=dark", wait_until="load")

        # Assert -- switcher is visible standalone (the inverse of the embedded case).
        expect(page.locator("osprey-theme-switcher")).to_be_visible()
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- click the switcher's toggle button.
        page.locator("#theme-toggle").click()

        # Assert -- toggleTheme() cycled dark -> light.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)

        page.close()


@pytest.mark.parametrize(
    ("panel_name", "launch", "path", "branding_selector"),
    _CHROME_CONTRACT_PANELS,
    ids=[name for name, _, _, _ in _CHROME_CONTRACT_PANELS],
)
def test_theme_toggle_strips_stale_query_param_and_survives_reload(
    panel_name, launch, path, branding_selector, tmp_path, monkeypatch, chromium_browser
):
    """D15: a toggle strips ``?theme=`` from the URL, so a reload can't resurrect it.

    Starts from a real ``?theme=dark`` query param (not merely its absence)
    so the post-toggle assertion proves setTheme()'s ``history.replaceState``
    strip actually removed something, rather than passing vacuously on a URL
    that never had the param to begin with.
    """
    del branding_selector  # unused here; shared parametrization with the embedded test above
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(f"{base_url}{path}?theme=dark", wait_until="load")
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- toggle via the switcher (the only path a follower ever
        # reaches setTheme() through).
        page.locator("#theme-toggle").click()

        # Assert -- the leftover ?theme=dark is gone from the URL immediately.
        assert "theme=" not in page.url

        # Act -- reload.
        page.reload(wait_until="load")

        # Assert -- the stale param can't be resurrected because it was
        # actually stripped (not just visually ignored): reload falls back
        # to OS/localStorage resolution, and the URL still carries no
        # ``theme=`` fragment for a future reload to trip over either.
        assert "theme=" not in page.url

        page.close()


def test_channel_finder_embedded_non_occlusion(tmp_path, monkeypatch, chromium_browser):
    """channel_finder's embedded mode keeps the fixed header visible without occluding content.

    D15 narrowed channel_finder's embedded-hide rule from the whole
    ``.app-header`` down to just ``.app-logo`` (css:118 -> ``.app-logo``,
    the old whole-header rule at :119 deleted) specifically so the pipeline
    switcher and nav stay usable inside the hub. This is the anti-regression
    check for that narrowing: the switcher must still render inside the
    viewport (not accidentally hidden or pushed off-screen), and
    ``.app-main``'s 48px top padding -- which exists so content clears the
    still-fixed 48px header -- must be unchanged in embedded mode. A future
    change that reintroduced whole-header hiding without also zeroing this
    padding would otherwise leave a silent 48px gap; one that hid the header
    without keeping this padding would occlude content start.
    """
    # Arrange
    with _launch_channel_finder(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # Assert -- the pipeline switcher is rendered and positioned inside the viewport.
        box = page.locator("#pipeline-switcher").bounding_box()
        assert box is not None, "#pipeline-switcher has no bounding box -- is it rendered?"
        viewport = page.viewport_size
        assert viewport is not None
        assert box["width"] > 0 and box["height"] > 0
        assert 0 <= box["y"] <= viewport["height"]

        # Assert -- .app-main's clearance for the still-fixed header is unchanged.
        padding_top = page.evaluate(
            "getComputedStyle(document.querySelector('.app-main')).paddingTop"
        )
        assert padding_top == "48px"

        page.close()
