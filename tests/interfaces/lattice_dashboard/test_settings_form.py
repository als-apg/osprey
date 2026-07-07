"""Browser Playwright interaction pin for the Lattice Dashboard settings form.

``test_osprey_drawer.py`` style (see
``tests/interfaces/design_system/test_osprey_drawer.py``): a real uvicorn
lattice_dashboard server over an empty ``tmp_path`` (mirrors
``test_load_smokes.py``'s ``_launch_lattice_dashboard``) driven by a real
chromium page via the shared ``chromium_browser`` fixture. This is the
per-cluster behavioral regression net for the computation-settings layer
(``static/js/settings.js``): it proves the real frontend JS, attached to the
real rendered DOM, drives the real backend correctly end to end — something
the settings.js/net.js Vitest unit suites (which mock fetch and never touch
a live server) cannot pin on their own.

Scope: open the settings tab, edit one field from each of the three
``SETTINGS_FIELDS`` type families (int, float, int_or_null), Apply, assert
the outgoing PUT payload shape and the SSE-driven re-rendered values; then a
separate Reset flow proves defaults come back.

Backend fidelity: ``/api/settings`` (GET/PUT/DELETE) and the SSE
``/api/events`` stream are real — no mocking — so the PUT's persistence and
its ``settings_updated`` broadcast (which is what actually re-renders the
form; ``applySettings()`` itself never touches the DOM after its PUT
resolves) are proven for real. The one exception is ``/api/refresh``, which
``applySettings()`` always calls after a successful PUT: on a real server
this spawns real ``sys.executable -m ...workers.<figure>`` subprocesses per
fast figure (``compute.py``'s ``_launch_worker``), which have no shutdown
hook wired into this app's lifespan and nothing to compute against without a
loaded lattice. That's a real, pre-existing, production-reachable code path,
just not one this settings-focused pin needs to exercise — so ``/api/refresh``
alone is routed to a canned 200, keeping the test fast and avoiding orphaned
worker processes, while everything settings-specific stays end to end.

Run:
    uv run pytest tests/interfaces/lattice_dashboard/test_settings_form.py -m browser -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Page, Request

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

SETTINGS_TAB_SELECTOR = '.sidebar-tab[data-tab="settings"]'
APPLY_BTN_SELECTOR = ".settings-btn--apply"
RESET_BTN_SELECTOR = ".settings-btn:not(.settings-btn--apply)"


# ---------------------------------------------------------------------------
# Live server launcher
# ---------------------------------------------------------------------------


@contextmanager
def _launch_lattice_dashboard(tmp_path, monkeypatch) -> Iterator[str]:
    """Real lattice_dashboard server over an empty ``tmp_path`` (mirrors test_load_smokes.py)."""
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.lattice_dashboard.app import create_app

    app = create_app(workspace_root=tmp_path)
    with _run_app_server(app) as base_url:
        yield base_url


def _mock_refresh(page: Page) -> None:
    """Short-circuit /api/refresh so Apply never spawns real worker subprocesses.

    Must be armed before navigation. See the module docstring for why this
    (and only this) endpoint is mocked.
    """
    page.route(
        "**/api/refresh",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status": "ok", "launched": []}',
        ),
    )


def _open_settings_tab(page: Page) -> None:
    """Click the sidebar's Settings tab (auto-expands a collapsed sidebar) and
    wait for the async GET /api/settings -> renderSettingsForm() round trip.
    """
    page.click(SETTINGS_TAB_SELECTOR)
    expect(page.locator("#tab-settings")).to_have_class(
        "sidebar-tab-content sidebar-tab-content--active"
    )
    expect(page.locator("#setting-da-nturns")).to_be_visible(timeout=5_000)


# ---------------------------------------------------------------------------
# Test 1: edit one field per type family, Apply, assert payload + re-render
# ---------------------------------------------------------------------------


def test_apply_sends_full_payload_and_rerenders_edited_values(
    tmp_path, monkeypatch, chromium_browser
):
    """Editing an int/float/int_or_null field and clicking Apply PUTs the
    whole form (every group, not just the edited fields) and the SSE
    settings_updated broadcast re-renders the form with the new values.
    """
    with _launch_lattice_dashboard(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _mock_refresh(page)
        put_requests: list[Request] = []
        page.on("request", lambda req: put_requests.append(req) if req.method == "PUT" else None)

        page.goto(base_url, wait_until="domcontentloaded")
        _open_settings_tab(page)

        # One field per SETTINGS_FIELDS type family, each a valid in-range
        # value so backend validation (_validate_setting) doesn't clamp it —
        # the re-render assertions below check for the exact value sent.
        int_field = page.locator("#setting-da-nturns")  # type: 'int'
        float_field = page.locator("#setting-chromaticity-dp_min_pct")  # type: 'float'
        nullable_field = page.locator("#setting-lma-n_sectors")  # type: 'int_or_null'

        int_field.fill("1024")
        int_field.press("Tab")
        float_field.fill("-12.5")
        float_field.press("Tab")
        nullable_field.fill("7")
        nullable_field.press("Tab")

        page.click(APPLY_BTN_SELECTOR)

        # Wait on the re-rendered values first: the settings_updated SSE
        # broadcast (which is what actually re-renders the form) only fires
        # after the backend has processed the PUT, so by the time these
        # succeed the PUT request has necessarily already landed in
        # put_requests below -- no separate poll needed for the request capture.
        expect(page.locator("#setting-da-nturns")).to_have_value("1024", timeout=5_000)
        expect(page.locator("#setting-chromaticity-dp_min_pct")).to_have_value("-12.5")
        expect(page.locator("#setting-lma-n_sectors")).to_have_value("7")

        # --- PUT payload shape ---
        assert len(put_requests) >= 1, "expected at least one PUT /api/settings request"
        payload = put_requests[0].post_data_json
        assert payload is not None, "PUT /api/settings should carry a JSON body"
        settings = payload["settings"]
        # collectSettingsFromForm() scans every rendered input, so the PUT
        # carries all four groups, not just the three edited fields.
        assert set(settings.keys()) == {"da", "lma", "chromaticity", "footprint"}
        assert settings["da"]["nturns"] == 1024
        assert settings["chromaticity"]["dp_min_pct"] == -12.5
        assert settings["lma"]["n_sectors"] == 7

        page.close()


# ---------------------------------------------------------------------------
# Test 2: Reset restores every group's declared defaults
# ---------------------------------------------------------------------------


def test_reset_restores_defaults(tmp_path, monkeypatch, chromium_browser):
    """After editing and applying a value away from its default, Reset
    restores it (and an untouched-but-still-checked field) to the backend's
    declared defaults.
    """
    with _launch_lattice_dashboard(tmp_path, monkeypatch) as base_url:
        page = chromium_browser.new_page()
        _mock_refresh(page)
        page.goto(base_url, wait_until="domcontentloaded")
        _open_settings_tab(page)

        nturns = page.locator("#setting-da-nturns")
        nturns.fill("2048")
        nturns.press("Tab")
        page.click(APPLY_BTN_SELECTOR)
        expect(nturns).to_have_value("2048", timeout=5_000)

        page.click(RESET_BTN_SELECTOR)

        # DEFAULT_SETTINGS (state.py): da.nturns=512, chromaticity.dp_min_pct=-3.0,
        # lma.n_sectors=None (rendered as an empty value with the "auto" placeholder).
        expect(page.locator("#setting-da-nturns")).to_have_value("512", timeout=5_000)
        expect(page.locator("#setting-chromaticity-dp_min_pct")).to_have_value("-3")
        n_sectors = page.locator("#setting-lma-n_sectors")
        expect(n_sectors).to_have_value("")
        expect(n_sectors).to_have_attribute("placeholder", "auto")

        page.close()
