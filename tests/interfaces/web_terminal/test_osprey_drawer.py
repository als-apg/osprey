"""Browser Playwright suite for web_terminal's ``<osprey-drawer>`` superset.

Exercises the opt-in features (B.1) as actually wired into web_terminal (B.2):
resizable, tabs, the unsaved-changes guard, and the settings-specific
first-time warning gate layered on top by ``settings.js`` — none of which
ariel's base-only migration (A.3/A.4, ``tests/interfaces/design_system/
test_osprey_drawer.py``) exercises. This is the hard parity + safety gate the
plan calls out: green here is required before the old ``drawer.js`` deletion
(already landed in B.2) is accepted as final.

Covers the acceptance criteria for B.2/B.3:
  - tabs switch (active tab + panel track clicks; ``drawer:tab-activate``
    fires on the newly active panel)
  - resize persists across reload (same ``osprey-drawer-width`` localStorage
    key) and clamps to the 320px/90vw bounds
  - the unsaved-changes guard blocks close via BOTH backdrop click and Escape,
    and releases once the guard allows it (safety-adjacent: this is the
    drawer that gates safety-hook config)
  - ``drawer:open``/``drawer:close``/``drawer:tab-activate`` reach all four
    consumer modules (scaffold-gallery, memory-gallery, settings, hook-debug)
  - the header trigger's ``.active`` class tracks open/close
  - the web_terminal skin (480px width, CRT glow, backdrop keyed off `[open]`)
    is unchanged
  - ``showSettingsWarning`` still gates the drawer once per server session,
    pinning the B.2 review's fixes: C1 (gate installs independent of
    `#tab-config`), M1 (a rapid second click never spawns a second dialog)

Follows the harness pattern from ``test_load_smokes.py``'s ``_launch_web_terminal``:
a real uvicorn web_terminal server on a free port, over an empty ``tmp_path``,
with the companion-backend spawns (artifact server, panel/web config loaders)
patched out exactly as the smoke suite does, and a real chromium page via the
shared ``chromium_browser`` fixture.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_osprey_drawer.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

# The settings trigger intentionally carries `data-drawer-trigger`, not the
# component's own `[data-drawer]` marker — see settings.js's module docstring
# on initSettingsWarningGate: this keeps the component's delegated handler
# from ever matching (and toggling) this button directly, so the warning
# gate below is the sole open path with no propagation tampering.
TRIGGER_SELECTOR = '[data-drawer-trigger="settings-drawer"]'
DRAWER_SELECTOR = "#settings-drawer"
BACKDROP_SELECTOR = "#drawer-backdrop"
CLOSE_BTN_SELECTOR = ".drawer-close-btn"
RESIZE_HANDLE_SELECTOR = ".drawer-resize-handle"
WARNING_PROCEED_SELECTOR = ".settings-warning-proceed"
WARNING_CANCEL_SELECTOR = ".settings-warning-cancel"
WARNING_OVERLAY_SELECTOR = ".settings-warning-overlay"
WELCOME_OVERLAY_SELECTOR = "#welcome-overlay"

VIEWPORT = {"width": 1280, "height": 800}


# ---------------------------------------------------------------------------
# Live server launcher (mirrors test_load_smokes.py's _launch_web_terminal)
# ---------------------------------------------------------------------------


@contextmanager
def _launch_web_terminal(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    with (
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

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


def _goto(page: Page, base_url: str) -> None:
    """Navigate to the hub and dismiss the first-visit welcome banner.

    ``app.js``'s ``initWelcomeModal()`` shows a full-screen overlay on every
    fresh (no prior localStorage ack) session that otherwise intercepts every
    click, including the settings gear -- unrelated to the drawer itself, but
    every test needs it out of the way first. Its Enter-to-dismiss keydown
    listener is only attached after its own internal ``await fetchJSON('/health')``
    resolves, so a single blind keypress races that attachment (it can lose,
    especially once a test's own ``page.route`` adds latency to every
    request) -- retry the press in a short bounded poll instead.
    """
    page.goto(base_url, wait_until="domcontentloaded")
    overlay = page.locator(WELCOME_OVERLAY_SELECTOR)
    if overlay.count() == 0:
        return
    deadline = time.monotonic() + 5.0
    while overlay.count() > 0 and time.monotonic() < deadline:
        page.keyboard.press("Enter")
        page.wait_for_timeout(100)
    expect(overlay).to_have_count(0, timeout=5_000)


def _open_settings_drawer(page: Page) -> None:
    """Click the real header trigger, proceeding past the first-time warning
    dialog if it appears (already-acked sessions skip straight to open) --
    the realistic end-to-end path, not a direct ``.open()`` API bypass.

    Also waits for the slide-in CSS transition (``transform: translateX``,
    250ms) to settle before returning: raw ``page.mouse`` drag simulations
    (unlike ``.click()``) skip Playwright's actionability/stability wait, so
    a resize drag started mid-transition targets a handle that has already
    moved by the time the drag's mousemove events fire.
    """
    page.click(TRIGGER_SELECTOR)
    proceed = page.locator(WARNING_PROCEED_SELECTOR)
    try:
        expect(proceed).to_be_visible(timeout=2_000)
        proceed.click()
    except AssertionError:
        pass  # already acked this server session -- drawer opened directly
    drawer = page.locator(DRAWER_SELECTOR)
    expect(drawer).to_have_attribute("open", "", timeout=5_000)
    expect(drawer).to_have_css("transform", "matrix(1, 0, 0, 1, 0, 0)", timeout=2_000)


def _drag_resize_handle(page: Page, dx: int) -> None:
    """Drag ``.drawer-resize-handle`` horizontally by ``dx`` pixels.

    The drawer is right-anchored, so dragging left (negative dx) grows it and
    dragging right (positive dx) shrinks it -- matching osprey-drawer.js's
    ``_beginResizeDrag`` (``dx = startX - moveEvent.clientX``), so a caller
    passing a *negative* dx here should widen the drawer.
    """
    handle = page.locator(RESIZE_HANDLE_SELECTOR)
    box = handle.bounding_box()
    assert box is not None, "resize handle has no bounding box -- is it rendered?"
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    # osprey-drawer.js computes width from clientX deltas on 'mousemove', not
    # from a single jump -- move in a couple of steps for a realistic drag.
    # Final mouse clientX = start_x + dx, so the component's own
    # `dx_js = startX - moveEvent.clientX` works out to exactly `-dx` here --
    # i.e. a negative `dx` (mouse moves left) widens, matching the docstring.
    page.mouse.move(start_x + dx / 2, start_y)
    page.mouse.move(start_x + dx, start_y)
    page.mouse.up()


# ---------------------------------------------------------------------------
# Test 1: tabs switch
# ---------------------------------------------------------------------------


def test_tabs_switch_updates_active_tab_and_panel(tmp_path, monkeypatch, chromium_browser):
    """Clicking a tab activates its button and panel, deactivating the others,
    and fires ``drawer:tab-activate`` on the newly active panel.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)

        # Behavior tab is active by default (markup).
        expect(page.locator('.drawer-tab[data-tab="tab-behavior"]')).to_have_class(
            "drawer-tab active"
        )
        expect(page.locator("#tab-behavior")).to_have_class("drawer-tab-panel active")

        page.evaluate(
            "document.getElementById('tab-safety')"
            ".addEventListener('drawer:tab-activate', () => { window.__tabActivateFired = true; })"
        )
        page.click('.drawer-tab[data-tab="tab-safety"]')

        expect(page.locator('.drawer-tab[data-tab="tab-safety"]')).to_have_class(
            "drawer-tab active"
        )
        expect(page.locator("#tab-safety")).to_have_class("drawer-tab-panel active")
        # Behavior deactivated -- exactly one tab/panel active at a time.
        expect(page.locator('.drawer-tab[data-tab="tab-behavior"]')).to_have_class("drawer-tab")
        expect(page.locator("#tab-behavior")).to_have_class("drawer-tab-panel")
        assert page.evaluate("window.__tabActivateFired") is True, (
            "expected drawer:tab-activate to fire on #tab-safety, the newly active panel"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 2: resize persists across reload
# ---------------------------------------------------------------------------


def test_resize_persists_width_across_reload(tmp_path, monkeypatch, chromium_browser):
    """Dragging the resize handle changes the width (persisted under the same
    ``osprey-drawer-width`` localStorage key as the old drawer.js); it
    survives a reload.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)

        drawer = page.locator(DRAWER_SELECTOR)
        initial_width = drawer.bounding_box()["width"]
        assert initial_width == pytest.approx(480, abs=1), (
            f"expected the default 480px web_terminal skin, got {initial_width}"
        )

        _drag_resize_handle(page, dx=-100)  # drag left -> widen
        widened_width = drawer.bounding_box()["width"]
        assert widened_width == pytest.approx(initial_width + 100, abs=2), (
            f"expected ~{initial_width + 100}px after the drag, got {widened_width}"
        )
        persisted = page.evaluate("localStorage.getItem('osprey-drawer-width')")
        assert persisted == str(round(widened_width)), (
            f"expected the drag to persist under the legacy 'osprey-drawer-width' key, "
            f"got {persisted!r}"
        )

        page.reload(wait_until="domcontentloaded")
        _open_settings_drawer(page)
        restored_width = page.locator(DRAWER_SELECTOR).bounding_box()["width"]
        assert restored_width == pytest.approx(widened_width, abs=2), (
            f"expected the persisted width ~{widened_width}px to survive reload, "
            f"got {restored_width}"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 3: resize clamps to the 320px / 90vw bounds
# ---------------------------------------------------------------------------


def test_resize_clamps_to_bounds(tmp_path, monkeypatch, chromium_browser):
    """Dragging past either bound clamps to 320px (floor) or 90vw (ceiling)."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)
        drawer = page.locator(DRAWER_SELECTOR)

        # Drag far right (shrinking) well past the 320px floor.
        _drag_resize_handle(page, dx=1000)
        floor_width = drawer.bounding_box()["width"]
        assert floor_width == pytest.approx(320, abs=1), (
            f"expected the 320px floor, got {floor_width}"
        )

        # Drag far left (widening) well past the 90vw ceiling (1280 * 0.9 = 1152).
        _drag_resize_handle(page, dx=-3000)
        ceiling_width = drawer.bounding_box()["width"]
        assert ceiling_width == pytest.approx(1152, abs=1), (
            f"expected the 90vw ceiling (1152px at {VIEWPORT['width']}px viewport), "
            f"got {ceiling_width}"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 4: unsaved-changes guard blocks close via backdrop AND Escape, and
# releases once the guard allows it
# ---------------------------------------------------------------------------


def test_unsaved_guard_blocks_close_via_backdrop_and_escape(
    tmp_path, monkeypatch, chromium_browser
):
    """A guard returning false blocks close via both the backdrop and Escape;
    once the guard allows it, the same paths close normally.

    Safety-adjacent: this drawer gates safety-hook configuration, so a
    pending edit must not be silently discardable either way.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)
        page.evaluate(
            "window.__allowClose = false;"
            "document.getElementById('settings-drawer')"
            ".registerUnsavedGuard(() => window.__allowClose)"
        )

        page.locator(BACKDROP_SELECTOR).click(position={"x": 10, "y": 10})
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=2_000)

        page.keyboard.press("Escape")
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=2_000)

        # The guard allows it now -- Escape (and by the same code path, the
        # backdrop) closes normally again.
        page.evaluate("window.__allowClose = true;")
        page.keyboard.press("Escape")
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 5: drawer:tab-activate reaches hook-debug (safety) and settings (config)
# ---------------------------------------------------------------------------


def test_tab_activate_reaches_hook_debug_and_settings(tmp_path, monkeypatch, chromium_browser):
    """Switching to the Safety/Config tabs fires each panel's own consumer request.

    Uses network requests rather than rendered content, since scaffold.js's
    artifact fetch is shared/cached across tabs (see test 6/7 below) while
    ``/api/hooks/debug-status`` (hook-debug.js) and ``/api/config``
    (settings.js) are each that consumer's own, uncached endpoint -- firing
    only if that module's own ``drawer:tab-activate`` listener ran.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        requests: list[str] = []
        page.on("request", lambda req: requests.append(req.url))
        _goto(page, base_url)

        _open_settings_drawer(page)

        page.click('.drawer-tab[data-tab="tab-safety"]')
        expect(page.locator("#tab-safety")).to_have_class("drawer-tab-panel active")
        assert any("/api/hooks/debug-status" in u for u in requests), (
            "expected hook-debug.js's drawer:tab-activate listener to fire on the Safety tab"
        )

        page.click('.drawer-tab[data-tab="tab-config"]')
        expect(page.locator("#tab-config")).to_have_class("drawer-tab-panel active")
        assert any("/api/config" in u for u in requests), (
            "expected settings.js's drawer:tab-activate listener to fire on the Config tab"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 6: drawer:open/close reach scaffold-gallery (behavior tab, default-active)
# ---------------------------------------------------------------------------


def test_drawer_open_reaches_scaffold_gallery_and_close_resets_it(
    tmp_path, monkeypatch, chromium_browser
):
    """Opening fires scaffold-gallery's load (Behavior is the default-active
    tab); closing resets it (and the shared artifact-fetch cache) so a later
    reopen refetches rather than silently reusing stale gallery state.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        requests: list[str] = []
        page.on("request", lambda req: requests.append(req.url))
        _goto(page, base_url)

        _open_settings_drawer(page)
        first_count = sum("/api/scaffold" in u for u in requests)
        assert first_count >= 1, (
            "expected scaffold-gallery.js's drawer:open (via the default-active "
            "Behavior tab's drawer:tab-activate) to fetch /api/scaffold"
        )

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)

        _open_settings_drawer(page)
        second_count = sum("/api/scaffold" in u for u in requests)
        assert second_count > first_count, (
            "expected drawer:close to reset the Behavior gallery and clear the shared "
            "fetch cache, so reopening refetches /api/scaffold instead of reusing "
            "stale (loaded=true, cached-promise) state"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 7: drawer:tab-activate/close reach memory-gallery
# ---------------------------------------------------------------------------


def test_tab_activate_and_close_reach_memory_gallery(tmp_path, monkeypatch, chromium_browser):
    """Switching to Memory fires memory-gallery's load; close resets it too."""
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        requests: list[str] = []
        page.on("request", lambda req: requests.append(req.url))
        _goto(page, base_url)

        _open_settings_drawer(page)
        page.click('.drawer-tab[data-tab="tab-memory"]')
        expect(page.locator("#tab-memory")).to_have_class("drawer-tab-panel active")
        first_count = sum("/api/claude-memory" in u for u in requests)
        assert first_count >= 1, (
            "expected memory-gallery.js's drawer:tab-activate listener to fetch "
            "/api/claude-memory on the Memory tab"
        )

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)

        # Reopening leaves Memory as the still-active tab (tab state itself
        # isn't reset by close, only gallery-internal .loaded/cache state is),
        # so it re-fires drawer:tab-activate on the same panel without another click.
        _open_settings_drawer(page)
        expect(page.locator("#tab-memory")).to_have_class("drawer-tab-panel active")
        second_count = sum("/api/claude-memory" in u for u in requests)
        assert second_count > first_count, (
            "expected drawer:close to reset the memory gallery and clear its fetch "
            "cache, so reopening refetches /api/claude-memory instead of reusing "
            "stale (loaded=true, cached-promise) state"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 8: header trigger .active tracks open/close
# ---------------------------------------------------------------------------


def test_active_trigger_highlight_tracks_open_close(tmp_path, monkeypatch, chromium_browser):
    """The gear button gains ``.active`` on open and loses it on close.

    app.js's initDrawerTriggerHighlight() wires this via the drawer:open/
    close events (osprey-drawer.js itself deliberately doesn't manage a
    trigger's .active state -- see its module docstring).
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        trigger = page.locator(TRIGGER_SELECTOR)
        expect(trigger).to_have_class("header-icon-btn")

        _open_settings_drawer(page)
        expect(trigger).to_have_class("header-icon-btn active")

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(trigger).to_have_class("header-icon-btn")
        page.close()


# ---------------------------------------------------------------------------
# Test 9: web_terminal skin unchanged (480px, CRT glow, backdrop keyed off [open])
# ---------------------------------------------------------------------------


def test_skin_matches_web_terminal_480px_with_glow(tmp_path, monkeypatch, chromium_browser):
    """The web_terminal skin (480px width, ``::before`` CRT glow) is intact,
    and the shared backdrop keys off the canonical ``[open]`` attribute state
    (not a class), matching the A.3 ariel-migration retarget pattern.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        _open_settings_drawer(page)

        width = page.locator(DRAWER_SELECTOR).bounding_box()["width"]
        assert width == pytest.approx(480, abs=1), (
            f"expected the 480px web_terminal skin, got {width}"
        )

        background_image = page.evaluate(
            """() => {
                const el = document.getElementById('settings-drawer');
                return getComputedStyle(el, '::before').backgroundImage;
            }"""
        )
        assert "gradient" in background_image, (
            f"expected the CRT-glow gradient on ::before, got {background_image!r}"
        )

        expect(page.locator(BACKDROP_SELECTOR)).to_have_attribute("open", "", timeout=2_000)
        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(page.locator(BACKDROP_SELECTOR)).not_to_have_attribute("open", "", timeout=2_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 10: showSettingsWarning gates the drawer, once per server session
# ---------------------------------------------------------------------------


def test_settings_warning_gates_first_open_and_acks_for_session(
    tmp_path, monkeypatch, chromium_browser
):
    """First open shows the warning and the drawer never has `[open]` until
    Proceed; Cancel leaves it closed; Proceed opens it and acknowledges, so a
    later open in the same server session skips the dialog entirely.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        # First click: the warning dialog appears, drawer stays closed.
        page.click(TRIGGER_SELECTOR)
        overlay = page.locator(WARNING_OVERLAY_SELECTOR)
        expect(overlay).to_be_visible(timeout=5_000)
        expect(page.locator(".settings-warning-title")).to_have_text("Expert Configuration Area")
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=2_000)

        # Cancel: dialog dismissed, drawer never opened.
        page.click(WARNING_CANCEL_SELECTOR)
        expect(overlay).not_to_be_visible(timeout=5_000)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=2_000)

        # Click again, this time Proceed: dialog dismissed, drawer opens, ack persisted.
        page.click(TRIGGER_SELECTOR)
        expect(overlay).to_be_visible(timeout=5_000)
        page.click(WARNING_PROCEED_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=5_000)

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)

        # Same server session, reopen: no warning dialog this time.
        page.click(TRIGGER_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=5_000)
        expect(overlay).not_to_be_visible()
        page.close()


# ---------------------------------------------------------------------------
# Test 11 (C1 regression pin): the gate installs independent of #tab-config
# ---------------------------------------------------------------------------


def test_settings_warning_gate_installs_without_tab_config(tmp_path, monkeypatch, chromium_browser):
    """The warning gate must install even if `#tab-config` is absent (C1).

    settings.js's initSettings() used to resolve `#tab-config` and bail
    before installing the gate if it was missing -- a fail-OPEN regression a
    reviewer caught (the gate must depend ONLY on the drawer + trigger
    existing). Pinned here at full page-load fidelity: the served index.html
    is rewritten so `document.getElementById('tab-config')` resolves to
    nothing, and the gate must still show the warning dialog rather than
    silently letting the component's own [data-drawer] toggle... except this
    trigger never carries that attribute in the first place (see the
    TRIGGER_SELECTOR comment above), so a fail-open here would instead mean
    no dialog and the drawer simply never opening at all on this click.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:

        def _strip_tab_config(route):
            if route.request.url.rstrip("/") != base_url.rstrip("/"):
                route.continue_()
                return
            response = route.fetch()
            body = response.text().replace('id="tab-config"', 'id="tab-config-removed-for-c1-test"')
            route.fulfill(response=response, body=body)

        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.route("**/*", _strip_tab_config)
        _goto(page, base_url)

        assert page.evaluate("document.getElementById('tab-config')") is None, (
            "test setup broken -- #tab-config should be absent from the served page"
        )

        page.click(TRIGGER_SELECTOR)
        expect(page.locator(WARNING_OVERLAY_SELECTOR)).to_be_visible(timeout=5_000)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=2_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 12 (M1 regression pin): a rapid second click never spawns a second dialog
# ---------------------------------------------------------------------------


def test_settings_warning_double_click_yields_one_dialog(tmp_path, monkeypatch, chromium_browser):
    """Two rapid gear clicks must show exactly one warning dialog -- pins the
    B.2 review's M1 fix (`warningGatePending`).

    Dispatches both clicks via a single ``page.evaluate()`` (two synchronous
    ``.click()`` calls in one JS turn) rather than two separate Playwright
    ``page.click()`` actions: once the first click's dialog is up, it is a
    full-screen overlay that legitimately covers the trigger, so a second
    *Playwright* click action would just get stuck retrying against real
    actionability semantics -- that isn't the race this test is after. Two
    native ``.click()`` calls back-to-back in one script guarantee both DOM
    click events fire before either handler's async continuation (or the
    resulting dialog) can interleave, which is precisely "rapid double
    click": JS's synchronous, in-dispatch-order event handling -- not network
    timing -- is what the `warningGatePending` fix relies on.
    """
    with _launch_web_terminal(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page(viewport=VIEWPORT)
        _goto(page, base_url)

        page.evaluate(
            f"""() => {{
                const trigger = document.querySelector('{TRIGGER_SELECTOR}');
                trigger.click();
                trigger.click();
            }}"""
        )

        expect(page.locator(WARNING_OVERLAY_SELECTOR)).to_have_count(1, timeout=5_000)
        page.close()
