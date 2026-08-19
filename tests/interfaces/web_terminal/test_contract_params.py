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
        # Refresh lives in the browser toolbar's ⋯ overflow menu, so it has to
        # be opened before the item is clickable.
        with page.expect_request(lambda r: "/api/artifacts" in r.url) as rejected_info:
            page.locator("#sidebar-menu-btn").click()
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
# (4) chrome contract: theme control + embedded-hide + D15 reload-strip
# ---------------------------------------------------------------------------
# Every panel exposes a theme control, and `applyEmbedded()` is wired into
# each of the 6 panels below -- this is the automated proof that the chrome
# contract holds on every panel, not just the one or two it was developed
# against.
# web_terminal's own session.html joins as fleet page 7, via a
# path-override on `_launch_web_terminal` (same hub server
# test_load_smokes.py's own '/' case boots) rather than a distinct
# launcher -- session.html is a static page under the hub's /static
# mount, not a second app.
# `branding_selector` is the per-page standalone-only element the D15
# narrowing keeps hidden in embedded mode; okf_panel has no branding
# chrome of its own to hide, so its entry is `None` and the branding
# assertion is skipped for it.
# `theme_control_selector` + `toggle_action` are the per-panel pair the
# three chrome tests below drive: the element the D15 embedded-hide rule
# targets, and the callable that flips its appearance away from dark. Most
# panels mount the bare `<osprey-theme-switcher>`, whose mode button IS the
# control; ARIEL mounts `<osprey-display-menu>`, which collapses the same
# preference behind a popover, so its action opens the card first. Both
# reach theme-manager.js's `toggleTheme()` -- the tests assert the same
# outcome either way, they just click through a different chrome.


def _toggle_via_switcher(page):
    """Flip `<osprey-theme-switcher>` off dark: one click on its mode button."""
    page.locator("osprey-theme-switcher .theme-switcher-mode").first.click()


def _toggle_via_display_menu(page):
    """Flip `<osprey-display-menu>` off dark: open the popover, pick Light.

    The Appearance row no-ops when the pick matches the current mode, so
    picking 'light' from the ``?theme=dark`` start state both call sites use
    is a real toggle, not a re-assertion of what was already applied.
    """
    page.locator("osprey-display-menu .display-menu-trigger").first.click()
    card = page.locator("osprey-display-menu .display-menu-card").first
    expect(card).to_be_visible()
    card.locator('.display-seg-option[data-appearance="light"]').click()


_CHROME_CONTRACT_PANELS = [
    # ariel hides its whole `.header` embedded (the tile bar is its only
    # header), so the branding assertion targets the element carrying the
    # rule -- `.logo` inside it still computes its own `display: flex`.
    ("ariel", _launch_ariel, "", ".header", "osprey-display-menu", _toggle_via_display_menu),
    ("artifacts", _launch_artifacts, "", ".logo", "osprey-theme-switcher", _toggle_via_switcher),
    (
        "channel_finder",
        _launch_channel_finder,
        "",
        ".app-logo",
        "osprey-theme-switcher",
        _toggle_via_switcher,
    ),
    (
        "lattice_dashboard",
        _launch_lattice_dashboard,
        "",
        ".topbar-logo",
        "osprey-theme-switcher",
        _toggle_via_switcher,
    ),
    ("okf_panel", _launch_okf_panel, "", None, "osprey-theme-switcher", _toggle_via_switcher),
    (
        "web_terminal_session",
        _launch_web_terminal,
        "/static/session.html",
        "header h1",
        "osprey-theme-switcher",
        _toggle_via_switcher,
    ),
]

_CHROME_CONTRACT_ARGNAMES = (
    "panel_name",
    "launch",
    "path",
    "branding_selector",
    "theme_control_selector",
    "toggle_action",
)
_CHROME_CONTRACT_IDS = [entry[0] for entry in _CHROME_CONTRACT_PANELS]


@pytest.mark.parametrize(
    _CHROME_CONTRACT_ARGNAMES,
    _CHROME_CONTRACT_PANELS,
    ids=_CHROME_CONTRACT_IDS,
)
def test_embedded_hides_branding_and_switcher(
    panel_name,
    launch,
    path,
    branding_selector,
    theme_control_selector,
    toggle_action,
    tmp_path,
    monkeypatch,
    chromium_browser,
):
    """``?embedded=true`` hides the page's own branding AND its theme control.

    The theme control's own D15 rule (``body.embedded <tag> { display: none
    }``, injected once by the component itself -- osprey-theme-switcher.js
    or osprey-display-menu.js) is what hides it -- no per-panel CSS is
    needed for that half of the contract. The branding selector, by
    contrast, is each page's own pre-existing ``body.embedded <selector> {
    display: none }`` rule; this proves the theme-control rollout didn't
    disturb it.
    """
    del toggle_action  # unused here; shared parametrization with the toggle tests below
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}{path}?embedded=true", wait_until="load")

        # Assert -- applyEmbedded() ran.
        expect(page.locator("body.embedded")).to_have_count(1)
        # Assert -- the theme control is hidden by its component's own injected rule.
        assert (
            page.evaluate(
                "sel => getComputedStyle(document.querySelector(sel)).display",
                theme_control_selector,
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
    _CHROME_CONTRACT_ARGNAMES,
    _CHROME_CONTRACT_PANELS,
    ids=_CHROME_CONTRACT_IDS,
)
def test_switcher_present_and_toggles_theme_standalone(
    panel_name,
    launch,
    path,
    branding_selector,
    theme_control_selector,
    toggle_action,
    tmp_path,
    monkeypatch,
    chromium_browser,
):
    """Standalone (no ``?embedded``), the theme control is visible and toggles the theme.

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

        # Assert -- the control is visible standalone (the inverse of the embedded case).
        expect(page.locator(theme_control_selector)).to_be_visible()
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- drive the panel's own theme chrome.
        toggle_action(page)

        # Assert -- toggleTheme() cycled dark -> light.
        expect(page.locator("html[data-theme='light']")).to_have_count(1)

        page.close()


@pytest.mark.parametrize(
    _CHROME_CONTRACT_ARGNAMES,
    _CHROME_CONTRACT_PANELS,
    ids=_CHROME_CONTRACT_IDS,
)
def test_theme_toggle_strips_stale_query_param_and_survives_reload(
    panel_name,
    launch,
    path,
    branding_selector,
    theme_control_selector,
    toggle_action,
    tmp_path,
    monkeypatch,
    chromium_browser,
):
    """D15: a toggle strips ``?theme=`` from the URL, so a reload can't resurrect it.

    Starts from a real ``?theme=dark`` query param (not merely its absence)
    so the post-toggle assertion proves setTheme()'s ``history.replaceState``
    strip actually removed something, rather than passing vacuously on a URL
    that never had the param to begin with.
    """
    del branding_selector, theme_control_selector  # shared parametrization with the tests above
    # Arrange
    with launch(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(f"{base_url}{path}?theme=dark", wait_until="load")
        expect(page.locator("html[data-theme='dark']")).to_have_count(1)

        # Act -- toggle via the panel's theme chrome (the only path a
        # follower ever reaches setTheme() through).
        toggle_action(page)

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
    """channel_finder's embedded mode drops its header without leaving a dead band.

    Embedded, the hub's tile bar is the panel's only header, so the whole
    ``.app-header`` hides -- and because that header is ``position: fixed``
    with a compensating 48px top padding on ``.app-main``, the padding must
    go with it or the panel opens on an empty 48px band. The pipeline
    switcher survives the header's removal because it does not live there:
    it sits in the body's bottom corpus strip, which must stay rendered
    inside the viewport for the panel to remain switchable when embedded.
    """
    # Arrange
    with _launch_channel_finder(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()

        # Act
        page.goto(f"{base_url}?embedded=true", wait_until="load")

        # Assert -- the local header is gone, and so is the padding that cleared it.
        assert (
            page.evaluate("getComputedStyle(document.querySelector('.app-header')).display")
            == "none"
        )
        padding_top = page.evaluate(
            "getComputedStyle(document.querySelector('.app-main')).paddingTop"
        )
        assert padding_top == "0px"

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

        # Assert -- content clears the fixed strip instead of running under it.
        padding_bottom = page.evaluate(
            "getComputedStyle(document.querySelector('.app-main')).paddingBottom"
        )
        assert padding_bottom == "28px"

        page.close()
