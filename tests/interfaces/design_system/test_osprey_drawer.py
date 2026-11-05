"""Browser Playwright suite for ``<osprey-drawer>``: base behavior + dialog a11y.

Exercises the shared drawer component
(``design_system/static/js/components/osprey-drawer.js``) against its first
real consumer, ariel's Settings drawer — the base-only migration with no
opt-in features (see A.3). Covers: custom-element upgrade; open/close via
trigger, close-button, backdrop click and Escape; single-active exclusivity;
and the dialog accessibility contract (focus enter/trap/restore, `role`/
`aria-modal`/`aria-labelledby`, background `inert`). web_terminal's superset
(resizable/tabs/guard) gets its own parity + safety suite once migrated (B.3,
``tests/interfaces/web_terminal/test_osprey_drawer.py``) — this file stays
scoped to the base contract, proven on the interface with zero coupling to
the superset features currently being added to the component in parallel.

Placement rationale: the component itself lives under ``design_system``, and
this package already owns the pattern for browser-proving design-system
modules against a live consumer app (see ``test_behavioral.py`` for
``theme-manager.js``); ariel here plays the same "real consumer, driven live"
role that the hub does there. Keeping this out of ``tests/interfaces/ariel/``
or ``tests/interfaces/web_terminal/`` also leaves the latter namespace clear
for B.3's parity suite, as directed.

Launches through ``test_load_smokes.py``'s ``launch_ariel``:
a real uvicorn ariel server on a free port, monkeypatched into an empty
``tmp_path`` (no ``config.yml`` — the Settings drawer's config fetch 404s,
which never touches the focusable-element set this suite's a11y assertions
rely on: the failure handler only rewrites a status label, it never calls
``renderConfigForm``) and a real chromium page via the shared
``chromium_browser`` fixture.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_osprey_drawer.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.interfaces._browser import assert_page_loads_clean
from tests.interfaces._launchers import launch_ariel

if TYPE_CHECKING:
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

# The Settings entry lives inside <osprey-display-menu>'s popover card, so
# reaching it means opening the popover first (see _open_settings_drawer).
MENU_TRIGGER_SELECTOR = "osprey-display-menu .display-menu-trigger"
TRIGGER_SELECTOR = '[data-drawer="settings-drawer"]'
DRAWER_SELECTOR = "#settings-drawer"
BACKDROP_SELECTOR = "#drawer-backdrop"
# Scoped to Settings: the Status drawer carries its own close button.
CLOSE_BTN_SELECTOR = "#settings-drawer .drawer-close-btn"
# Last focusable descendant in initial DOM order (see the drawer markup in
# ariel/static/index.html): the raw-YAML mode button is the last enabled,
# visible, focusable element — the Apply button starts `disabled`, and both
# the raw editor and the confirm dialog start hidden (`display: none`) until
# interacted with, which excludes their descendants via `offsetParent`.
LAST_FOCUSABLE_SELECTOR = '.settings-mode-btn[data-mode="raw"]'


def _open_settings_drawer(page: Page) -> None:
    """Open the display-menu popover, click its Settings entry, await the drawer."""
    page.click(MENU_TRIGGER_SELECTOR)
    page.click(TRIGGER_SELECTOR)
    expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=5_000)


# ---------------------------------------------------------------------------
# Test 1: custom-element upgrade + clean load
# ---------------------------------------------------------------------------


def test_custom_element_upgrades_and_page_loads_clean(tmp_path, monkeypatch, chromium_browser):
    """``settings-drawer`` upgrades to the real class; the page loads with no error."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        assert_page_loads_clean(page, base_url)

        is_upgraded = page.evaluate(
            """() => {
                const el = document.getElementById('settings-drawer');
                const ctor = customElements.get('osprey-drawer');
                return !!el && !!ctor && el instanceof ctor;
            }"""
        )
        assert is_upgraded, "settings-drawer did not upgrade to the osprey-drawer class"
        page.close()


# ---------------------------------------------------------------------------
# Test 2: open via trigger, close via close button
# ---------------------------------------------------------------------------


def test_open_via_trigger_and_close_via_close_button(tmp_path, monkeypatch, chromium_browser):
    """The popover's Settings entry opens the drawer; the close button closes it."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 3: close via backdrop click
# ---------------------------------------------------------------------------


def test_close_via_backdrop_click(tmp_path, monkeypatch, chromium_browser):
    """Clicking the shared backdrop closes the open drawer."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        # Click a corner of the full-viewport backdrop clearly outside the
        # right-anchored 560px drawer, so the click actually lands on the
        # backdrop rather than the drawer sitting above it in stacking order.
        page.locator(BACKDROP_SELECTOR).click(position={"x": 10, "y": 10})
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(page.locator(BACKDROP_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 4: close via Escape
# ---------------------------------------------------------------------------


def test_close_via_escape_key(tmp_path, monkeypatch, chromium_browser):
    """Pressing Escape while a drawer is open closes it."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        page.keyboard.press("Escape")
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 5: single-active exclusivity
# ---------------------------------------------------------------------------


def test_single_active_exclusivity(tmp_path, monkeypatch, chromium_browser):
    """Opening a second drawer closes any other open drawer (module-level exclusivity)."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        # Inject a second, independent <osprey-drawer> and open it
        # programmatically — proves exclusivity holds for any drawer pair,
        # not just a hardcoded second instance in the page's own markup.
        page.evaluate(
            """() => {
                const el = document.createElement('osprey-drawer');
                el.id = 'second-drawer';
                document.body.appendChild(el);
                el.open();
            }"""
        )

        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(page.locator("#second-drawer")).to_have_attribute("open", "", timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 6: focus enters, is trapped, and is restored on close
# ---------------------------------------------------------------------------


def test_focus_enters_traps_and_restores_on_close(tmp_path, monkeypatch, chromium_browser):
    """Opening moves focus in and traps Tab; closing restores focus to the trigger."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        # Focus enters: the first focusable descendant is the close button.
        expect(page.locator(CLOSE_BTN_SELECTOR)).to_be_focused(timeout=5_000)

        # Shift+Tab from the first focusable wraps to the last.
        page.keyboard.press("Shift+Tab")
        expect(page.locator(LAST_FOCUSABLE_SELECTOR)).to_be_focused(timeout=5_000)

        # Tab from the last focusable wraps forward back to the first.
        page.keyboard.press("Tab")
        expect(page.locator(CLOSE_BTN_SELECTOR)).to_be_focused(timeout=5_000)

        # Closing restores focus to the display-menu trigger: the Settings
        # entry that opened the drawer sits inside the popover card, which is
        # display:none by now, so the component hands focus to its persistent
        # trigger before the drawer records its return-focus target.
        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(page.locator(MENU_TRIGGER_SELECTOR)).to_be_focused(timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 7: dialog ARIA semantics
# ---------------------------------------------------------------------------


def test_dialog_aria_semantics(tmp_path, monkeypatch, chromium_browser):
    """The open drawer exposes role=dialog/aria-modal and a resolvable accessible name."""
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")

        _open_settings_drawer(page)

        drawer = page.locator(DRAWER_SELECTOR)
        expect(drawer).to_have_attribute("role", "dialog")
        expect(drawer).to_have_attribute("aria-modal", "true")
        expect(drawer).to_have_attribute("tabindex", "-1")

        labelledby = drawer.get_attribute("aria-labelledby")
        assert labelledby, "expected aria-labelledby to resolve to the drawer's title"
        title_text = page.evaluate(
            "(id) => document.getElementById(id)?.textContent?.trim()", labelledby
        )
        assert title_text == "Settings", (
            f"aria-labelledby resolved to unexpected text: {title_text!r}"
        )
        page.close()


# ---------------------------------------------------------------------------
# Test 8: background inert while open, including pre-existing overlays
# ---------------------------------------------------------------------------


# A one-pixel PNG: the lightbox only needs an image response, not a picture.
_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c63f8cfc0f01f0005000201e221bc33"
    "0000000049454e44ae426082"
)
_ENTRY_ID = "E-LIGHTBOX"
_ATTACHMENT_ID = "A-LIGHTBOX"


def _route_entry_with_picture(page: Page) -> None:
    """Serve one entry with one viewable picture, and the picture itself."""
    entry = {
        "entry_id": _ENTRY_ID,
        "source_system": "test",
        "timestamp": "2026-01-01T12:00:00Z",
        "author": "tester",
        "subject": "Entry with a picture",
        "raw_text": "See the attached picture.",
        "attachments": [
            {
                "attachment_id": _ATTACHMENT_ID,
                "filename": "beam.png",
                "type": "image/png",
                "viewable": True,
                "display_url": f"/attachments/{_ATTACHMENT_ID}",
            }
        ],
        "keywords": [],
        "metadata": {},
    }
    page.route(f"**/api/entries/{_ENTRY_ID}", lambda route: route.fulfill(json=entry))
    page.route(
        f"**/attachments/{_ATTACHMENT_ID}",
        lambda route: route.fulfill(body=_PIXEL_PNG, content_type="image/png"),
    )


def test_background_inert_while_open_including_preexisting_modal(
    tmp_path, monkeypatch, chromium_browser
):
    """While the drawer is open, every other top-level body child is inert.

    Includes ariel's image lightbox, a pre-existing top-level overlay that is
    not a drawer — intended dialog semantics (only one modal-like surface may
    be interactive at a time), confirmed during the component's A.2 review.
    ``osprey-drawer`` never inerts another drawer, so the overlay has to be a
    plain body child. The lightbox (z-index 10000) covers the header, so the
    drawer is opened through the store rather than a click that would land on
    the scrim and dismiss the lightbox.
    """
    with launch_ariel(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _route_entry_with_picture(page)
        page.goto(
            f"{base_url}/#entry?id={_ENTRY_ID}&attachment={_ATTACHMENT_ID}",
            wait_until="domcontentloaded",
        )

        lightbox = page.locator("#image-lightbox")
        expect(lightbox).to_be_visible(timeout=10_000)

        page.evaluate("import('/static/js/store.js').then(m => m.setState({ drawer: 'settings' }))")
        expect(page.locator(DRAWER_SELECTOR)).to_have_attribute("open", "", timeout=5_000)

        app_container = page.locator(".app-container")
        expect(app_container).to_have_attribute("inert", "", timeout=5_000)
        expect(app_container).to_have_attribute("aria-hidden", "true")
        expect(lightbox).to_have_attribute("inert", "", timeout=5_000)
        expect(lightbox).to_have_attribute("aria-hidden", "true")

        page.click(CLOSE_BTN_SELECTOR)
        expect(page.locator(DRAWER_SELECTOR)).not_to_have_attribute("open", "", timeout=5_000)
        expect(app_container).not_to_have_attribute("inert", "", timeout=5_000)
        expect(lightbox).not_to_have_attribute("inert", "", timeout=5_000)
        expect(lightbox).not_to_have_attribute("aria-hidden", "true")
        page.close()
