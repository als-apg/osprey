"""Browser suite: the dispatch dashboard's panel toolbar and its ⋯ entry.

The EVENTS panel has one 40 px toolbar: an Activity | Triggers segmented switch
at the start and one search field at the end. The search field searches runs on
Activity and filters triggers by name on Triggers. The hub's tile bar gets one
v3 ``menu`` contribution, whose "Clear filters" entry resets the window, the
status filter, the lane and both searches, and the status sentence.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_dispatch_dashboard_toolbar.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces.design_system._dispatch_dashboard import create_dispatch_dashboard_app

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1100, "height": 900}

TRIGGERS = ("hello-dispatch", "triage-event", "save-report")


def _trigger(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "armed",
        "source": "webhook",
        "on_error": "drop",
        "allowed_tools": [],
        "prompt": "p",
        "source_config": {},
        "last_fired": None,
        "next_fire": None,
    }


def _state(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pool": {"running": 0, "queued": 0, "max": 2},
        "triggers": [_trigger(n) for n in TRIGGERS],
        "runs": runs,
        "timeline": {},
        "worker_error": None,
        "server_time_iso": None,
        "facility_timezone": None,
    }


RUNS = [
    {
        "run_id": "aaaaaaaa-1",
        "trigger_name": "hello-dispatch",
        "status": "completed",
        "age_sec": 30,
    },
    {"run_id": "bbbbbbbb-2", "trigger_name": "triage-event", "status": "error", "error": "boom"},
]


@pytest.fixture
def busy_url() -> Iterator[str]:
    """Three triggers and two runs, one of them failed."""
    with run_app_server(create_dispatch_dashboard_app(_state(RUNS))) as base_url:
        yield base_url


@pytest.fixture
def quiet_url() -> Iterator[str]:
    """Three triggers and no runs yet."""
    with run_app_server(create_dispatch_dashboard_app(_state([]))) as base_url:
        yield base_url


def _open(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelector('#panel-state').hidden",
        timeout=10_000,
    )


def _pressed(page: Page, seg: str) -> list[str]:
    return page.eval_on_selector_all(
        f"{seg} > button[aria-pressed='true']", "bs => bs.map(b => b.textContent.trim())"
    )


def test_the_segment_switches_views_and_a_detail_keeps_its_parent_pressed(
    chromium_browser: Browser, busy_url: str
) -> None:
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, f"{busy_url}/")
    assert _pressed(page, "#view-seg") == ["Activity"]
    assert page.is_visible("#view-activity")

    page.click("#view-seg button[data-view='triggers']")
    assert _pressed(page, "#view-seg") == ["Triggers"]
    assert page.is_visible("#view-triggers")
    assert not page.is_visible("#view-activity")

    page.click("#trigger-list .trigger-row")
    assert page.is_visible("#view-trigger")
    assert _pressed(page, "#view-seg") == ["Triggers"]
    assert page.is_disabled("#search")
    assert "hello-dispatch" in page.inner_text("#toolbar-crumb")

    # The pressed parent segment leads back to its list.
    page.click("#view-seg button[data-view='triggers']")
    assert page.is_visible("#view-triggers")
    assert page.is_hidden("#toolbar-crumb")
    page.close()


def test_the_search_filters_runs_on_activity_and_triggers_on_triggers(
    chromium_browser: Browser, busy_url: str
) -> None:
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, f"{busy_url}/")
    page.wait_for_selector("#run-list .run-row")
    page.fill("#search", "triage")
    page.wait_for_function("() => document.querySelectorAll('#run-list .run-row').length === 1")
    assert "triage-event" in page.inner_text("#run-list")

    page.click("#view-seg button[data-view='triggers']")
    # Each list keeps its own query: the run search does not carry over.
    assert page.input_value("#search") == ""
    assert page.get_attribute("#search", "placeholder") == "Filter triggers…"
    page.fill("#search", "save")
    page.wait_for_function(
        "() => document.querySelectorAll('#trigger-list .trigger-row').length === 1"
    )
    assert "save-report" in page.inner_text("#trigger-list")

    page.click("#view-seg button[data-view='activity']")
    assert page.input_value("#search") == "triage"
    page.close()


def test_the_section_filters_are_segs_and_the_empty_list_offers_view_triggers(
    chromium_browser: Browser, quiet_url: str
) -> None:
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open(page, f"{quiet_url}/")
    assert _pressed(page, "#timeline-window") == ["24h"]
    assert _pressed(page, "#status-filter") == ["All"]

    state = page.locator("#run-list .osprey-state")
    assert "No runs recorded yet." in state.inner_text()
    state.locator("button", has_text="View triggers").click()
    assert page.is_visible("#view-triggers")
    assert _pressed(page, "#view-seg") == ["Triggers"]
    page.close()


def test_the_tile_bar_gets_the_menu_and_the_status_and_clear_filters_resets(
    chromium_browser: Browser, busy_url: str
) -> None:
    """Embedded, the panel contributes one v3 ``menu`` and its status; the entry resets every filter."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    # A same-origin host page standing in for the hub: it frames the panel and
    # records each contribution.
    page.goto(f"{busy_url}/dashboard/state", wait_until="domcontentloaded")
    page.evaluate(
        """() => {
          window.__items = [];
          addEventListener('message', (e) => {
            if (e.data?.type === 'canopy-host-header-contribution') window.__items.push(e.data.items);
          });
          const f = document.createElement('iframe');
          f.id = 'panel';
          f.style.cssText = 'width:1000px;height:800px';
          f.src = '/?embedded=true&status=error&hours=1&q=boom';
          document.body.append(f);
        }"""
    )
    page.wait_for_function("() => window.__items.length > 0", timeout=10_000)
    kinds = page.evaluate("() => [...new Set(window.__items.flat().map(i => i.kind))]")
    last = page.evaluate("() => window.__items.at(-1)")
    assert kinds == ["menu", "text"]
    assert last[0] == {
        "kind": "menu",
        "id": "events",
        "items": [{"id": "clear-filters", "label": "Clear filters", "disabled": False}],
    }
    assert "1 failed" in last[1]["text"]

    frame = page.frame_locator("#panel")
    assert frame.locator("#status-filter > button[aria-pressed='true']").inner_text() == "Failed"

    page.evaluate(
        """() => document.getElementById('panel').contentWindow.postMessage(
          { type: 'canopy-host-header-action', id: 'events', value: 'clear-filters' },
          location.origin)"""
    )
    page.wait_for_function(
        "() => window.__items.at(-1)[0].items[0].disabled === true", timeout=10_000
    )
    assert frame.locator("#status-filter > button[aria-pressed='true']").inner_text() == "All"
    assert frame.locator("#timeline-window > button[aria-pressed='true']").inner_text() == "24h"
    assert frame.locator("#search").input_value() == ""
    search = page.evaluate("() => document.getElementById('panel').contentWindow.location.search")
    assert search == "?embedded=true"
    page.close()
