"""Browser smoke tests: SSE → DOM for dynamic agent panels.

Proves three frontend behaviors that the FastAPI TestClient cannot reach because
it doesn't run a real browser or a live SSE stream:

  CC-1  hide-active switches / empty-states
  CF-2  register adds a tab without rebuilding existing tabs (no renderTabs call)
  CC-3  register does NOT auto-activate the new tab

Each test launches a real uvicorn server in a background thread, drives events
through the REST API, and asserts the DOM via Playwright auto-waiting.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_panels_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
import requests

from tests.interfaces.conftest import _run_app_server

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Live-server context manager
# ---------------------------------------------------------------------------

# Custom panel used by tests that need a second, immediately-healthy tab.
# healthEndpoint=None → panel becomes healthy immediately (no health polling).
# The raw URL is stored server-side; the browser receives /panel/data-viz.
_CUSTOM_DATA_VIZ: dict = {
    "id": "data-viz",
    "label": "DATA VIZ",
    "url": "http://data-viz.internal:8080",
    "healthEndpoint": None,
    "path": "/",
}


@contextmanager
def _live_server(workspace_dir, enabled_panels, custom_panels=None, allow_runtime: bool = False):
    """Launch a real web terminal server on a free port in a background thread.

    Companion backends (artifact server, ARIEL, etc.) are bypassed via patches
    so no external process dependencies are required.  The artifacts panel
    reports its URL as http://127.0.0.1:8086 (the standard fallback).

    Yields:
        (base_url: str, app: FastAPI) — live server address and the FastAPI app
        object for post-startup state manipulation.
    """
    if custom_panels is None:
        custom_panels = []

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels, None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_artifact_server",
            # Set the artifact URL without actually spawning a server process.
            side_effect=lambda a: setattr(a.state, "artifact_server_url", "http://127.0.0.1:8086"),
        ),
    ):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        # _run_app_server yields only after the port accepts connections (lifespan
        # done), so app.state is safe to mutate here; and it joins the server
        # thread while the patches above are still live, avoiding timing races.
        with _run_app_server(app) as base_url:
            if allow_runtime:
                app.state.allow_runtime_panels = True
                app.state.runtime_panel_allowlist = None  # any non-loopback host allowed

            yield base_url, app


# ---------------------------------------------------------------------------
# Shared helper: open page and wait for tabs to render
# ---------------------------------------------------------------------------


def _open_page(browser, base_url: str) -> Page:
    """Open a new browser page and wait for the artifacts tab to appear.

    panel-manager.js renders tabs asynchronously (it fetches /api/panels then
    each panel's config endpoint before building the DOM).  This helper blocks
    until the artifacts tab is present so individual tests can assume a stable
    starting DOM.
    """
    page = browser.new_page()
    page.goto(base_url, wait_until="domcontentloaded")
    # Artifacts is always enabled and the DEFAULT_PANEL_FALLBACK, so its tab
    # should appear quickly after the async init path completes.
    # Iframes also carry data-panel-id, so target the button element specifically.
    expect(page.locator('button[data-panel-id="artifacts"]')).to_be_attached(timeout=10_000)
    # The first-visit welcome overlay intercepts pointer events; remove it so
    # tests that click header controls get a genuinely interactable starting DOM.
    page.evaluate("document.getElementById('welcome-overlay')?.remove()")
    return page


# ---------------------------------------------------------------------------
# Test 1: visibility SSE hides and shows a tab (no reload)
# ---------------------------------------------------------------------------


def test_visibility_hide_and_show(tmp_path, chromium_browser):
    """SSE panel_visibility toggles .tab-hidden so the tab appears and disappears.

    Arrange: both artifacts and data-viz are visible at page load.
    Act: POST hide → POST show via /api/panel-visibility.
    Assert: tab transitions hidden→visible without a page reload.
    """
    # Arrange
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Target the tab BUTTON specifically; the iframe also carries data-panel-id.
        data_viz_tab = page.locator('button[data-panel-id="data-viz"]')

        # data-viz has no healthEndpoint → immediately healthy+enabled; tab visible
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        # Act — broadcast hide event via the API
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200

        # Assert — SSE adds .tab-hidden (display:none), so tab is no longer visible
        expect(data_viz_tab).not_to_be_visible(timeout=5_000)

        # Act — broadcast show event
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": True},
        )
        assert r.status_code == 200

        # Assert — .tab-hidden removed, tab visible again
        expect(data_viz_tab).to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 2: hiding the last visible+healthy panel renders the empty state (CC-1)
# ---------------------------------------------------------------------------


def test_hide_last_visible_panel_shows_empty_state(tmp_path, chromium_browser):
    """CC-1: hiding the active tab with no fallback shows 'No panels visible'.

    Arrange: only artifacts is enabled (no custom panels), so it is the sole
    visible+healthy tab and is auto-activated as DEFAULT_PANEL.
    Act: POST hide artifacts → SSE broadcast.
    Assert: panel-manager clears activeTabId and renders the empty state message.
    """
    # Arrange
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # Artifacts must be active (auto-activated as DEFAULT_PANEL) before we hide it.
        artifacts_active = page.locator('button[data-panel-id="artifacts"].active')
        expect(artifacts_active).to_be_attached(timeout=10_000)

        # Act — hide the only visible+healthy panel
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "artifacts", "visible": False},
        )
        assert r.status_code == 200

        # Assert — empty state message appears inside #panel-content (CC-1)
        panel_content = page.locator("#panel-content")
        expect(panel_content.get_by_text("No panels visible")).to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 3: register adds a tab non-destructively; no auto-activation (CF-2/CC-3)
# ---------------------------------------------------------------------------


def test_register_adds_tab_non_destructively(tmp_path, chromium_browser):
    """CF-2/CC-3: panel_register appends a tab; active tab is intact; not auto-activated.

    CF-2: addPanel() appends exactly one <button> to #header-tabs WITHOUT calling
          renderTabs() (which does innerHTML='' and destroys live state).
    CC-3: the newly registered panel is NOT auto-activated — activateTab() is not
          called, so the new tab never gets the 'active' CSS class.

    SSRF note: _validate_panel_url is patched to return None (pass) so this
    test can focus on the SSE→DOM wiring rather than DNS resolution. URL
    validation is already covered by the unit tests in test_proxy_panel.py.
    """
    # Arrange
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        # artifacts must be active before registration
        artifacts_active = page.locator('button[data-panel-id="artifacts"].active')
        expect(artifacts_active).to_be_attached(timeout=10_000)

        # Record DOM state: count tabs so we can verify exactly one was appended.
        initial_tab_count = page.locator(".header-tab").count()

        # Act — register a new panel; patch SSRF validation (async) to always pass
        with patch(
            "osprey.interfaces.web_terminal.routes.panels._validate_panel_url",
            new=AsyncMock(return_value=None),
        ):
            r = requests.post(
                f"{base_url}/api/panels/register",
                json={
                    "id": "monitor",
                    "label": "MONITOR",
                    "url": "http://grafana.internal:3000",
                    "path": "/",
                },
            )
        assert r.status_code == 200, r.text

        # Assert CF-2 — exactly one new tab appended (no duplicate, no rebuild)
        # Iframes also carry data-panel-id; target the button specifically.
        monitor_tab = page.locator('button[data-panel-id="monitor"]')
        expect(monitor_tab).to_be_visible(timeout=5_000)
        final_tab_count = page.locator(".header-tab").count()
        assert final_tab_count == initial_tab_count + 1, (
            f"Expected exactly one new tab; tab delta was {final_tab_count - initial_tab_count}"
        )

        # Assert CF-2 cont. — artifacts tab still active (renderTabs() clears 'active')
        expect(artifacts_active).to_be_attached(timeout=2_000)

        # Assert CC-3 — monitor tab is present but NOT active
        expect(page.locator('button[data-panel-id="monitor"]:not(.active)')).to_be_attached(
            timeout=2_000
        )

        page.close()


# ---------------------------------------------------------------------------
# Test 4: the "+" menu lists a hidden panel and reveals it (human add)
# ---------------------------------------------------------------------------


def test_add_menu_reveals_hidden_panel(tmp_path, chromium_browser):
    """Clicking a panel in the "+" menu un-hides its tab.

    Arrange: data-viz starts visible, then is hidden via the API so it becomes a
    known-but-hidden panel — exactly what the "+" menu offers under "Show panel".
    Act: open the "+" menu and click the DATA VIZ entry.
    Assert: the tab is visible again (the menu drove POST /api/panel-visibility).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        # Scope to the tab strip: once the "+" menu opens, its menu item also
        # carries data-panel-id="data-viz", so target the header tab specifically.
        data_viz_tab = page.locator('button.header-tab[data-panel-id="data-viz"]')
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        # Hide it so it becomes a known-but-hidden panel the "+" menu can offer.
        r = requests.post(
            f"{base_url}/api/panel-visibility",
            json={"panel": "data-viz", "visible": False},
        )
        assert r.status_code == 200
        expect(data_viz_tab).not_to_be_visible(timeout=5_000)

        # Act — open the "+" menu; the hidden panel is listed as an add target.
        page.locator("#panel-add-btn").click()
        menu_item = page.locator('.panel-add-item[data-panel-id="data-viz"]')
        expect(menu_item).to_be_visible(timeout=5_000)
        menu_item.click()

        # Assert — the tab is revealed again.
        expect(data_viz_tab).to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 5: the per-tab "×" hides a panel (human remove)
# ---------------------------------------------------------------------------


def test_close_button_hides_panel(tmp_path, chromium_browser):
    """Clicking a tab's "×" hides it without activating the tab.

    Act: hover the data-viz tab to reveal its "×", then click it.
    Assert: the tab becomes hidden (the "×" drove POST /api/panel-visibility).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_tab = page.locator('button[data-panel-id="data-viz"]')
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        # Act — reveal and click the "×" inside the tab.
        data_viz_tab.hover()
        page.locator('button[data-panel-id="data-viz"] .tab-close').click()

        # Assert — SSE echo added .tab-hidden, so the tab is gone.
        expect(data_viz_tab).not_to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 6: the URL row is gated by web.allow_runtime_panels
# ---------------------------------------------------------------------------


def test_add_menu_url_row_hidden_when_disabled(tmp_path, chromium_browser):
    """With allow_runtime_panels off (default), the "+" menu shows no URL input."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        page.locator("#panel-add-btn").click()
        expect(page.locator(".panel-add-menu.open")).to_be_visible(timeout=5_000)
        # No "new panel from URL" affordance when runtime registration is disabled.
        expect(page.locator(".panel-add-input")).to_have_count(0)
        page.close()


def test_add_menu_url_row_shown_when_enabled(tmp_path, chromium_browser):
    """With allow_runtime_panels on, the "+" menu offers the URL input."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        page.locator("#panel-add-btn").click()
        expect(page.locator('.panel-add-menu input[name="url"]')).to_be_visible(timeout=5_000)
        page.close()


# ---------------------------------------------------------------------------
# Test 7: submitting the URL form registers a panel end-to-end (human add-URL)
# ---------------------------------------------------------------------------


def test_add_menu_registers_url_panel(tmp_path, chromium_browser):
    """Filling the URL form and clicking Add appends a new tab.

    SSRF validation is patched to pass (URL validation is unit-tested elsewhere);
    this exercises the human form → POST /api/panels/register → SSE → DOM path.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        with patch(
            "osprey.interfaces.web_terminal.routes.panels._validate_panel_url",
            new=AsyncMock(return_value=None),
        ):
            page.locator("#panel-add-btn").click()
            page.locator('.panel-add-menu input[name="url"]').fill("http://grafana.internal:3000")
            page.locator('.panel-add-menu input[name="label"]').fill("Monitor")
            page.locator(".panel-add-submit").click()

            # id derives from the label → "monitor"; the register SSE adds the tab.
            monitor_tab = page.locator('button[data-panel-id="monitor"]')
            expect(monitor_tab).to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 8: re-showing a panel after the empty state rebuilds its iframe
# ---------------------------------------------------------------------------


def test_reshow_after_empty_state_rebuilds_iframe(tmp_path, chromium_browser):
    """Closing the only panel then reopening it must not leave a blank pane.

    Regression: renderEmptyState does contentEl.innerHTML=... which detaches the
    cached iframe; activateTab must rebuild it (isConnected guard) rather than
    re-show the orphaned node, otherwise the pane is stuck on the empty state.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        artifacts_tab = page.locator('button.header-tab[data-panel-id="artifacts"]')
        expect(artifacts_tab).to_be_visible(timeout=10_000)

        # Close the only panel → empty state.
        artifacts_tab.hover()
        page.locator('button[data-panel-id="artifacts"] .tab-close').click()
        expect(page.locator("#panel-content").get_by_text("No panels visible")).to_be_visible(
            timeout=5_000
        )

        # Reopen it from the "+" menu.
        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-item[data-panel-id="artifacts"]').click()

        # The content iframe is rebuilt and the empty state is gone (not blank).
        expect(page.locator('iframe.panel-iframe[data-panel-id="artifacts"]')).to_be_attached(
            timeout=5_000
        )
        expect(page.locator("#panel-content").get_by_text("No panels visible")).to_have_count(0)

        page.close()


# ---------------------------------------------------------------------------
# Test 9: Delete on a focused tab closes it (keyboard parity for the "×")
# ---------------------------------------------------------------------------


def test_keyboard_delete_closes_focused_tab(tmp_path, chromium_browser):
    """Pressing Delete on a focused tab hides it (the "×" is mouse-only)."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)
        data_viz_tab = page.locator('button.header-tab[data-panel-id="data-viz"]')
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        data_viz_tab.focus()
        data_viz_tab.press("Delete")

        expect(data_viz_tab).not_to_be_visible(timeout=5_000)

        page.close()


# ---------------------------------------------------------------------------
# Test 10: a rejected URL registration surfaces the server's reason
# ---------------------------------------------------------------------------


def test_add_menu_shows_register_error(tmp_path, chromium_browser):
    """A register 4xx (here: built-in id collision) is shown inline in the menu."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[],
        allow_runtime=True,
    ) as (base_url, _app):
        page = _open_page(chromium_browser, base_url)

        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-menu input[name="url"]').fill("http://x.internal:3000")
        # Label "Lattice" derives id "lattice", which collides with a built-in →
        # the server 422s and the menu must display the reason, not fail silently.
        page.locator('.panel-add-menu input[name="label"]').fill("Lattice")
        page.locator(".panel-add-submit").click()

        error = page.locator(".panel-add-url-error")
        expect(error).to_be_visible(timeout=5_000)
        expect(error).to_contain_text("built-in")

        page.close()


# ---------------------------------------------------------------------------
# Test 11: applying a preset ("Layout") is exclusive — show members, hide rest
# ---------------------------------------------------------------------------


def test_apply_preset_shows_members_hides_rest(tmp_path, chromium_browser):
    """Clicking a Layout shows exactly its members, hides non-members, focuses first.

    Arrange: artifacts + data-viz both visible; a "Just viz" preset lists only
    data-viz. app.state.panel_presets is seeded before page load so /api/panels
    carries it.
    Act: open the "+" menu, click the "Just viz" Layout item.
    Assert: data-viz becomes the sole visible+active tab, artifacts is hidden,
    and NO blank "No panels visible" state appears — the show-before-hide ordering
    focuses the member before the non-member is hidden (a blank-state flash here
    is the documented trigger to escalate to a batch endpoint).
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[_CUSTOM_DATA_VIZ],
    ) as (base_url, app):
        # Seed presets before the page loads so panel-manager reads them at init.
        app.state.panel_presets = [
            {"name": "Just viz", "panels": ["data-viz"]},
            {"name": "Both", "panels": ["artifacts", "data-viz"]},
        ]
        page = _open_page(chromium_browser, base_url)

        artifacts_tab = page.locator('button.header-tab[data-panel-id="artifacts"]')
        data_viz_tab = page.locator('button.header-tab[data-panel-id="data-viz"]')
        expect(artifacts_tab).to_be_visible(timeout=10_000)
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        # Act — open the "+" menu and click the "Just viz" Layout.
        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-menu button.panel-add-item:has-text("Just viz")').click()

        # Assert — data-viz is the sole visible tab and is active; artifacts hidden.
        expect(data_viz_tab).to_be_visible(timeout=5_000)
        expect(page.locator('button.header-tab[data-panel-id="data-viz"].active')).to_be_attached(
            timeout=5_000
        )
        expect(artifacts_tab).not_to_be_visible(timeout=5_000)

        # Assert — no blank state was left behind (show-before-hide ordering).
        expect(page.locator("#panel-content").get_by_text("No panels visible")).to_have_count(0)

        page.close()


# ---------------------------------------------------------------------------
# Test 12: a preset whose first member is offline focuses a healthy member
# ---------------------------------------------------------------------------


def test_apply_preset_offline_first_member_focuses_healthy(tmp_path, chromium_browser):
    """A preset primary that is unreachable must not strand focus or blank the pane.

    The first member's backend is down (health poll fails → its tab stays
    disabled), so a blind focus on it would no-op. applyPreset falls through to
    the first HEALTHY member (data-viz); 'No panels visible' must never appear.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    offline = {
        "id": "offline-panel",
        "label": "OFFLINE",
        "url": "http://127.0.0.1:59999",  # nothing listening → health poll fails
        "healthEndpoint": "/health",
        "path": "/",
    }
    with _live_server(
        workspace,
        enabled_panels={"artifacts"},
        custom_panels=[offline, _CUSTOM_DATA_VIZ],
    ) as (base_url, app):
        # Primary member is the offline panel; data-viz is the healthy fallback.
        app.state.panel_presets = [
            {"name": "Diag", "panels": ["offline-panel", "data-viz"]},
        ]
        page = _open_page(chromium_browser, base_url)

        data_viz_tab = page.locator('button.header-tab[data-panel-id="data-viz"]')
        expect(data_viz_tab).to_be_visible(timeout=10_000)

        page.locator("#panel-add-btn").click()
        page.locator('.panel-add-menu button.panel-add-item:has-text("Diag")').click()

        # The healthy member becomes active (the offline primary cannot), and the
        # non-member artifacts tab is hidden — with no blank pane in between.
        expect(page.locator('button.header-tab[data-panel-id="data-viz"].active')).to_be_attached(
            timeout=5_000
        )
        expect(page.locator('button.header-tab[data-panel-id="artifacts"]')).not_to_be_visible(
            timeout=5_000
        )
        expect(page.locator("#panel-content").get_by_text("No panels visible")).to_have_count(0)

        page.close()
