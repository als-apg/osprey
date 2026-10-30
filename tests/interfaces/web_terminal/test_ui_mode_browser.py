"""Browser tests: the Expert/Simple UI-mode mechanic, end to end.

Proves the mode axis (``data-ui-mode`` on ``<html>``) behaves correctly across
the pre-paint resolution ladder, the More menu's View rows, and reloads — behavior the
FastAPI TestClient cannot reach because none of it exists until a real browser
runs ``mode-boot.js`` in ``<head>`` and ``app.js``'s toggle wiring afterwards.

Coverage (one test each):

  (a) a hub whose ``web.ui_mode`` is ``simple`` first-paints Simple with NO
      Expert flash — the pre-paint ladder's SSR ``data-ui-mode`` rung.
  (b) the artifacts viewer embedded in a Simple-mode hub first-paints in Simple
      mode with NO Expert flash — the ``?mode=`` rung, stamped onto the iframe
      URL by panel-manager next to ``?theme=``.
  (c) picking a View row in the sidebar's More menu persists to localStorage and
      survives a reload.
  (d) toggling in a tab opened with ``?mode=simple`` still sticks after reload —
      the ``history.replaceState`` strip stops the spent URL param from
      out-ranking the fresh explicit choice.
  (e) ``?mode=`` out-ranks a conflicting localStorage value.
  (f) the sidebar — search and the More menu — renders in both modes, and the
      More menu's View rows report the mode they would pick, across a live flip.

"No flash" is asserted at the earliest observable point, not just on the settled
DOM: an init script installed at document-start records the value of
``data-ui-mode`` the instant the document element exists and every time it
mutates, so the assertion is that ``"expert"`` was *never* observed — a proxy for
"the wrong shell never painted", since ``mode-boot.js`` runs synchronously ahead
of every stylesheet.  That probe only observes pre-paint on a *top-level*
navigation (Playwright injects init scripts into a JS-created iframe only after
it has already loaded, too late to see its first paint), so the artifacts case
proves pre-paint by loading the exact URL the hub stamped onto the iframe as its
own page, and separately asserts the live embed inside the hub renders Simple.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_ui_mode_browser.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces._hub import hub_server
from tests.interfaces._panel_launch import publish_artifact_url

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import FrameLocator

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Pre-paint observation: record every data-ui-mode value from document-start
# ---------------------------------------------------------------------------
#
# Installed via add_init_script so it runs before any page script, including
# mode-boot.js.  ``document`` always exists at document-start but the document
# ELEMENT may not yet, so it attaches the moment ``<html>`` is inserted (the
# earliest reliable point), snapshots the SSR value there, then a
# MutationObserver appends every later value.  A correct boot yields either
# ["simple"] (hub: SSR already simple, mode-boot no-clobbers) or [null, "simple"]
# (artifacts: no SSR attr, mode-boot sets it from ?mode=) — never "expert".
_MODE_LOG_INIT_SCRIPT = """
(function () {
  window.__uiModeLog = window.__uiModeLog || [];
  function attach(el) {
    window.__uiModeLog.push(el.getAttribute('data-ui-mode'));
    new MutationObserver(function () {
      window.__uiModeLog.push(el.getAttribute('data-ui-mode'));
    }).observe(el, { attributes: true, attributeFilter: ['data-ui-mode'] });
  }
  if (document.documentElement) {
    attach(document.documentElement);
  } else {
    new MutationObserver(function (_m, obs) {
      if (document.documentElement) { obs.disconnect(); attach(document.documentElement); }
    }).observe(document, { childList: true, subtree: true });
  }
})();
"""

_ARTIFACTS_IFRAME = 'iframe.panel-iframe[data-panel-id="artifacts"]'


# ---------------------------------------------------------------------------
# Live-server: a real hub in front of a real embedded artifacts backend
# ---------------------------------------------------------------------------


@contextmanager
def _hub_with_artifacts(tmp_path: Path, ui_mode: str) -> Iterator[tuple[str, object]]:
    """Launch a hub (in ``ui_mode``) pointed at a genuinely reachable artifacts app.

    Mirrors ``test_visual.py``'s hub launcher (the same three patches around
    ``create_app`` plus a real artifacts backend so the embedded iframe loads
    real markup instead of a proxy error page), but yields the FastAPI ``app``
    too and pins the server-rendered UI mode.

    ``web.ui_mode`` reaches SSR through ``app.state.web_ui_mode``, which the
    ``GET /`` handler reads per request; the lifespan seeds it from config, and
    we override it post-startup — the same post-startup ``app.state`` seam the
    panels browser suite uses for ``visible_panels``/``panel_presets``.  This is
    the value ``index.html`` stamps onto ``<html data-ui-mode>``.

    Yields:
        (base_url, app) — the hub address and its FastAPI app.
    """
    from osprey.interfaces.artifacts.app import create_app as create_artifacts_app

    workspace = tmp_path / "hub_ws"
    workspace.mkdir(exist_ok=True)
    # One real artifact: a Simple hub with an EMPTY workspace boots chat-only
    # (workspace suppressed — see panel-manager.js), so the artifacts iframe
    # these tests observe would never mount. Mode propagation THROUGH the
    # workspace is the subject here, so the workspace must be present.
    (workspace / "seed_artifact.txt").write_text("hub-with-artifacts seed\n")

    with run_app_server(create_artifacts_app(workspace_root=workspace)) as artifact_url:
        with hub_server(workspace, launch=publish_artifact_url(artifact_url)) as (base_url, app):
            # Override the lifespan-resolved mode; root() re-reads it per request.
            app.state.web_ui_mode = ui_mode
            yield base_url, app


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _open_hub_page(browser: Browser, base_url: str, query: str = "") -> Page:
    """Open a fresh-context hub page with the pre-paint probe armed.

    ``browser.new_page()`` mints a new browser context, so localStorage never
    leaks between tests.  The probe is registered before navigation; the page
    then waits for the artifacts tab (async panel init done).
    """
    page = browser.new_page()
    page.add_init_script(_MODE_LOG_INIT_SCRIPT)
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)
    return page


def _assert_never_expert(page: Page, context: str) -> None:
    """The pre-paint probe must never have observed the Expert shell."""
    log = page.evaluate("window.__uiModeLog || []")
    assert "expert" not in log, f"{context}: Expert flash observed; data-ui-mode log was {log!r}"
    assert log and log[-1] == "simple", (
        f"{context}: expected the mode to settle on 'simple'; log was {log!r}"
    )


def _stored_mode(page: Page) -> str | None:
    return page.evaluate("localStorage.getItem('osprey-ui-mode')")


def _open_more_menu(page: Page) -> None:
    """Open the sidebar's More menu and wait for it to show."""
    page.locator("#more-btn").click()
    expect(page.locator("#more-menu")).to_be_visible()


def _close_more_menu(page: Page) -> None:
    """Dismiss the More menu if a pick left it open."""
    menu = page.locator("#more-menu")
    if menu.is_visible():
        page.keyboard.press("Escape")
    expect(menu).to_be_hidden()


def _view_row(page: Page, mode: str):
    return page.locator(f'#more-menu [role="menuitemradio"][data-view="{mode}"]')


def _click_segment(page: Page, mode: str) -> None:
    # The View rows live in the sidebar's More menu; open it, pick, and return
    # callers to the closed state whether or not the pick dismissed the menu.
    _open_more_menu(page)
    _view_row(page, mode).click()
    _close_more_menu(page)


# ---------------------------------------------------------------------------
# (a) SSR simple → hub boots Simple, no Expert flash
# ---------------------------------------------------------------------------


def test_simple_config_boots_simple_no_flash(tmp_path, chromium_browser):
    """A hub configured ``web.ui_mode: simple`` first-paints Simple with no flash.

    The SSR ``data-ui-mode="simple"`` rung is honored pre-paint: the probe must
    never see "expert" and the html attribute settles on "simple".
    """
    with _hub_with_artifacts(tmp_path, ui_mode="simple") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url)

        # No Expert flash at any observed point, and settled on Simple.
        _assert_never_expert(page, "hub first paint")
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")

        page.close()


# ---------------------------------------------------------------------------
# (b) artifacts iframe in a Simple hub first-paints Simple, no Expert flash
# ---------------------------------------------------------------------------


def _assert_artifact_viewer(root: Page | FrameLocator) -> None:
    """The embedded artifacts page is the hub's viewer of one artifact.

    The preview pane shows; the browser column is hidden (the hub's sidebar
    owns the list); there is no separate Simple section and no preview header
    (the hub's tile header names the artifact).
    """
    expect(root.locator("#browse-preview-pane")).to_be_visible(timeout=10_000)
    expect(root.locator("#browse-sidebar")).to_be_hidden()
    expect(root.locator("#view-artifacts-simple")).to_have_count(0)
    expect(root.locator(".preview-header")).to_have_count(0)


def test_artifacts_iframe_boots_simple_no_flash(tmp_path, chromium_browser):
    """The embedded artifacts viewer first-paints in Simple mode, no flash.

    Two halves, because a JS-created iframe's pre-paint is not observable from
    the parent (Playwright arms an iframe's init script only after it loads):

    1. On the live hub, panel-manager stamps ``mode=simple`` (read off the hub's
       ``<html>``) onto the iframe URL next to ``theme=``, and the embed carries
       ``data-ui-mode="simple"`` and renders the viewer: the preview pane, with
       no browser column, no Simple section and no preview header.
    2. Loading that exact stamped URL as its own page proves the artifacts
       document — which carries no SSR ``data-ui-mode`` — resolves Simple from
       the ``?mode=`` rung before first paint: the probe never sees "expert".
    """
    with _hub_with_artifacts(tmp_path, ui_mode="simple") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url)

        # artifacts is the DEFAULT_PANEL and healthy (real backend), so it
        # auto-activates and its iframe mounts + loads without interaction.
        expect(page.locator(_ARTIFACTS_IFRAME)).to_be_attached(timeout=10_000)
        iframe_src = page.locator(_ARTIFACTS_IFRAME).get_attribute("src")
        assert iframe_src and "mode=simple" in iframe_src, (
            f"hub must stamp mode=simple onto the artifacts iframe; src was {iframe_src!r}"
        )

        # Live embed: Simple mode, and the page is the viewer of one artifact.
        iframe = page.frame_locator(_ARTIFACTS_IFRAME)
        expect(iframe.locator("html")).to_have_attribute("data-ui-mode", "simple", timeout=10_000)
        _assert_artifact_viewer(iframe)
        page.close()

        # Pre-paint proof: load the exact stamped URL top-level, where the probe
        # can observe first paint. mode-boot resolves Simple from ?mode= with no
        # Expert ever on the element, and the viewer is what renders.
        probe = chromium_browser.new_page()
        probe.add_init_script(_MODE_LOG_INIT_SCRIPT)
        probe.goto(iframe_src, wait_until="domcontentloaded")
        expect(probe.locator("html")).to_have_attribute("data-ui-mode", "simple", timeout=10_000)
        _assert_artifact_viewer(probe)
        _assert_never_expert(probe, "artifacts first paint (stamped URL)")
        probe.close()


# ---------------------------------------------------------------------------
# (c) toggle persists to localStorage and survives reload
# ---------------------------------------------------------------------------


def test_toggle_persists_across_reload(tmp_path, chromium_browser):
    """Clicking Simple writes localStorage and the choice survives a reload.

    SSR stays Expert, so on reload the localStorage rung is what carries the
    Simple choice past the SSR default.
    """
    with _hub_with_artifacts(tmp_path, ui_mode="expert") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url)
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert")

        _click_segment(page, "simple")
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        assert _stored_mode(page) == "simple"

        page.reload(wait_until="domcontentloaded")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)

        # localStorage('osprey-ui-mode') out-ranks the Expert SSR default.
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")

        page.close()


# ---------------------------------------------------------------------------
# (d) a toggle overrides the spent ?mode= param after reload (replaceState strip)
# ---------------------------------------------------------------------------


def test_toggle_beats_spent_query_param_after_reload(tmp_path, chromium_browser):
    """Toggling in a ``?mode=simple`` tab sticks after reload.

    Opening with ``?mode=simple`` boots Simple.  Clicking Expert persists Expert
    AND strips ``?mode=`` from the URL via ``history.replaceState``.  Without the
    strip, the reload would still carry ``?mode=simple`` — the top ladder rung —
    and silently revert the operator's explicit Expert choice.  With it, the URL
    is clean and localStorage's Expert wins.
    """
    with _hub_with_artifacts(tmp_path, ui_mode="expert") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url, query="?mode=simple")
        # ?mode= is the top rung, so it boots Simple despite the Expert SSR default.
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")

        _click_segment(page, "expert")
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert")
        assert _stored_mode(page) == "expert"
        # The strip ran: the spent param is gone from the address bar.
        assert "mode=" not in page.url, f"?mode= should have been stripped; url was {page.url!r}"

        page.reload(wait_until="domcontentloaded")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)

        # The stripped param can no longer out-rank the stored Expert choice.
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert")

        page.close()


# ---------------------------------------------------------------------------
# (e) ?mode= out-ranks a conflicting localStorage value
# ---------------------------------------------------------------------------


def test_query_param_beats_localstorage(tmp_path, chromium_browser):
    """A live ``?mode=`` param wins over a conflicting stored mode.

    First establish a stored Simple choice via the toggle; then navigate the
    same context to ``?mode=expert``.  The query rung sits above the localStorage
    rung, so Expert must win on that load.
    """
    with _hub_with_artifacts(tmp_path, ui_mode="expert") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url)
        _click_segment(page, "simple")
        assert _stored_mode(page) == "simple"

        # Same context (localStorage persists), now with an explicit ?mode=expert.
        page.goto(f"{base_url}?mode=expert", wait_until="domcontentloaded")
        expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)

        # Query rung beats the stored 'simple'.
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "expert")
        assert _stored_mode(page) == "simple", (
            "the stored choice must be untouched by a ?mode= boot"
        )

        page.close()


# ---------------------------------------------------------------------------
# (f) the sidebar is the same in both modes, and the More menu names the mode
# ---------------------------------------------------------------------------


def test_sidebar_and_more_menu_are_mode_invariant(tmp_path, chromium_browser):
    """Search and ☰ More render in both modes; the View rows track the mode.

    The sidebar is the operator's fixed handhold in Expert and in Simple, so a
    live flip (no reload) leaves it in place. The More menu's View rows are a
    radio pair: the row for the current mode is the checked one.
    """
    with _hub_with_artifacts(tmp_path, ui_mode="expert") as (base_url, _app):
        page = _open_hub_page(chromium_browser, base_url)
        search = page.locator("#command-palette-btn")
        more = page.locator("#more-btn")

        expect(search).to_be_visible()
        expect(more).to_be_visible()
        _open_more_menu(page)
        expect(_view_row(page, "expert")).to_have_attribute("aria-checked", "true")
        expect(_view_row(page, "simple")).to_have_attribute("aria-checked", "false")
        _close_more_menu(page)

        _click_segment(page, "simple")
        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        expect(search).to_be_visible()
        expect(more).to_be_visible()
        _open_more_menu(page)
        expect(_view_row(page, "simple")).to_have_attribute("aria-checked", "true")
        expect(_view_row(page, "expert")).to_have_attribute("aria-checked", "false")
        _close_more_menu(page)

        page.close()
