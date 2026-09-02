"""Browser tests: the two bars an operator arranges, driven for real.

The header and the status bar are item HOSTS, and the whole point of the
feature is that the operator rearranges them by hand. Almost none of that is
observable from a FastAPI ``TestClient``: the drag is a pointer gesture with
``setPointerCapture`` and document-level listeners, the drop target is decided
from live ``getBoundingClientRect()`` boxes, the refusals are sentences painted
into a sheet, edit mode exists only after ``bar-customize.js`` has booted, and
"no horizontal overflow" is a statement about flex arithmetic that only a
layout engine can settle. The unit suites (``js/bar-customize*.test.js``,
``test_bar_items_routes.py``, ``test_bar_items_ssr.py``) pin every judgment in
isolation; this is the suite that proves the judgments are wired to a pointer
and a paint.

Coverage (one test each):

  (a) a tile dragged out of the Customize sheet lands in the bar it was
      dropped on, at the position it was dropped at — ONCE, which is the half
      that caught the stray click a finished drag leaves behind.
  (b) a drag inside one bar reorders it — the permutation case, which
      ``dropRefusal`` deliberately exempts from the full refusal ladder.
  (c) a drag across the two bars moves the item, leaving the source bar. Both
      directions, including into a bar whose last item never folds — the one
      that caught edit mode's decoration reporting crowding the bar did not have.
  (d) a drag that releases over NEITHER bar removes the item.
  (e) the arrangement survives a reload — it is stored server-side (``PUT
      /api/bar-items``), not in the tab, so the second paint is server-rendered
      from the same document.
  (f) no item is exempt from (d): the wordmark released outside the bars is
      removed like any other.
  (g) a header item dragged onto the status bar moves there — every type may
      sit in either bar — and a type this deployment cannot render is a
      disabled tile carrying its reason.
  (h) Simple mode renders the same saved arrangement and offers NO way into
      edit mode — no right-click menu, no display-menu row, no palette action —
      each absence paired with its presence in Expert on a cold page, because
      an entry point that mounts in neither mode reads the same as a gated one.
  (i) a header packed to the per-host cap with max-width gaps does not
      overflow a 1024 px viewport, the header chrome stays on screen, and the
      run's non-shrinking content leaves a stated margin rather than merely
      fitting. This is the declared ``flex`` hints doing the work, not a JS
      ladder rung.
  (j) the clock is LIVE on a real page: it reads ``HH:MM`` through the real
      boot path, not a hand-driven builder.
  (k) a deployment with no identity block paints no separator after the
      wordmark — the ``[data-follows]`` middot is keyed on the identity item,
      so a spacer inheriting the logo's follower slot must stay bare.
  (l) the "Default" preset discards the stored document (``DELETE``), the file
      leaves the store, and the bars come back as the deployment renders them.
  (m) a hidden header comes back from the terminal tile header's right-click
      menu -- the header has no surface of its own to right-click once hidden,
      so the tile headers carry the way back, and the restore is stored.

Fixtures follow ``test_osprey_drawer.py``'s ``_launch_web_terminal`` — a real
uvicorn web_terminal on a free port with the companion-backend spawns patched
out — plus one patch of its own: ``resolve_shared_data_root`` is re-pointed at
``tmp_path`` so the layouts these tests save land in a throwaway store. That is
the seam ``test_bar_items_routes.py`` uses and for the reason its own fixture
gives: ``OSPREY_AGENT_DATA_ROOT`` is not consulted by the resolver, so a test
trusting it writes into the repository's ``var/agent_data``.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_bar_items_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import requests

from osprey.interfaces.web_terminal.app import BAR_LAYOUT_VERSION
from osprey.interfaces.web_terminal.bar_items_store import LAYOUT_FILENAME
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from playwright.sync_api import Browser, Locator, Page

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

HEADER_HOST = '[data-bar-host="header"]'
STATUS_HOST = '[data-bar-host="status"]'
#: A shell the client has hydrated. `data-bar-key` is stamped by bar-host.js on
#: its first reconcile and never by the template, so its presence is the one
#: honest "the bar modules have booted" signal a test can wait on. Waiting on
#: `.bar-item` alone would pass against the server's first paint, before
#: anything is armed, and every gesture below would then land on nothing.
HYDRATED_SHELL = f"{HEADER_HOST} .bar-item[data-bar-key]"
SHEET = ".bar-sheet"
SHEET_NOTICE = ".bar-sheet-notice"
CONTEXT_MENU = ".bar-context-menu"
CUSTOMIZE_ROW = ".bar-customize-entry"
PALETTE_BTN = "#command-palette-btn"
PALETTE_INPUT = ".command-palette-input"
PALETTE_ITEM = ".command-palette-item"
DISPLAY_MENU_BTN = "#display-menu .display-menu-trigger"
DISPLAY_MENU_CARD = "#display-menu .display-menu-card"

VIEWPORT = {"width": 1280, "height": 800}
NARROW_VIEWPORT = {"width": 1024, "height": 768}

#: How much of a packed bar must still be free of NON-SHRINKING content. The
#: overflow test measures a real margin rather than just "it fits", because it
#: is the one assertion in this suite that depends on font metrics: this runner
#: is macOS, CI is Linux headless chromium, and the same system font stack is
#: wider there. A quarter is deliberately a wide bound — the measured figure is
#: around 0.59 — so it fails on a structural regression (a gap that stopped
#: shrinking, chrome that doubled) rather than on a few pixels of font drift.
MIN_BAR_HEADROOM_RATIO = 0.25


# ---------------------------------------------------------------------------
# Live server
# ---------------------------------------------------------------------------


@contextmanager
def _launch_web_terminal(tmp_path: Path) -> Iterator[tuple[str, Any]]:
    """A real web_terminal over a throwaway workspace and a throwaway store.

    Yields:
        ``(base_url, app)`` — the address and the FastAPI app, so a test can
        re-point the per-request ``app.state`` facts ``root()`` reads (the UI
        mode, the deployment name) the way the sibling browser suites do.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    agent_data = tmp_path / "agent_data"
    agent_data.mkdir(exist_ok=True)

    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
        # The store seam. Patched at the resolver, never through
        # OSPREY_AGENT_DATA_ROOT -- see test_bar_items_routes.py's fixture.
        patch(
            "osprey.utils.workspace.resolve_shared_data_root",
            return_value=agent_data,
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url, app


@pytest.fixture
def store_dir(tmp_path) -> Path:
    """Where ``_launch_web_terminal`` sites this test's layout document."""
    return tmp_path / "agent_data" / "bar_items"


# ---------------------------------------------------------------------------
# Seeding the stored arrangement
# ---------------------------------------------------------------------------


def _items(types: list[str | dict]) -> list[dict]:
    return [{"type": entry} if isinstance(entry, str) else entry for entry in types]


def _seed_layout(
    base_url: str,
    *,
    header: list[str | dict],
    status: list[str | dict],
    header_visible: bool = True,
    status_visible: bool = True,
) -> dict:
    """Store an arrangement through the real route, before the page is opened.

    Uses the HTTP contract rather than writing the file, so a document these
    tests build is one the store actually accepts -- a seed the server would
    refuse is a test that proves nothing about a browser. ``rev`` is 0 because
    the store is per-test and empty.
    """
    body = {
        "version": BAR_LAYOUT_VERSION,
        "rev": 0,
        "header": _items(header),
        "status": _items(status),
        "header_visible": header_visible,
        "status_visible": status_visible,
    }
    response = requests.put(f"{base_url}/api/bar-items", json=body, timeout=10)
    assert response.status_code == 200, f"seed refused: {response.status_code} {response.text}"
    return response.json()


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------

#: Seeded before every load: the onboarding tour is already dismissed. Under the
#: default `once` policy its invite card scrims the shell on the fresh profile
#: these tests run under and intercepts every click and press below. Copied from
#: test_osprey_drawer.py / test_scaffold_detail.py, which need it for the same
#: reason; the tour has its own coverage in tour.test.mjs.
_DISMISS_TOUR = "try { localStorage.setItem('osprey-tour-dismissed-v1', '1') } catch (e) {}"


def _open(browser: Browser, base_url: str, viewport: dict | None = None) -> Page:
    """A fresh-context page on the hub, booted, with the tour out of the way."""
    page = browser.new_page(viewport=viewport or VIEWPORT)
    page.add_init_script(_DISMISS_TOUR)
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector(HYDRATED_SHELL, timeout=15_000)
    return page


def _types(page: Page, host: str) -> list[str]:
    """The item types in a bar, in render order.

    Direct children only: the overflow ladder parks folded items in a menu
    outside the run, and a descendant query would report them as still placed.

    Read from ``data-bar-item``, the attribute the whole item model keys on and
    the one spelling of an item's type: the SSR template, ``bar-host.js`` and
    ``bars.css`` all use it, so a shell the client builds for an item dragged in
    at runtime is indistinguishable here from one the server rendered.
    """
    selector = f'[data-bar-host="{host}"] > .bar-item[data-bar-item]'
    return page.eval_on_selector_all(selector, "els => els.map((el) => el.dataset.barItem)")


def _center(locator: Locator) -> tuple[float, float]:
    box = locator.bounding_box()
    assert box is not None, "element has no bounding box -- is it rendered and visible?"
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _drag(page: Page, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Drive a pointer drag from *start* to *end*.

    ``page.mouse`` rather than ``drag_and_drop``: the gesture is built on
    pointer events with ``setPointerCapture`` and document-level
    pointermove/pointerup, and Playwright's drag helper drives the HTML5
    drag-and-drop protocol, which this build deliberately does not use.

    Several intermediate moves, not one jump: the first crosses the 4 px
    threshold that turns a press into a drag, and the rest are what
    ``aim()`` reads to decide the host and the insertion index.
    """
    page.mouse.move(*start)
    page.mouse.down()
    steps = 8
    for step in range(1, steps + 1):
        page.mouse.move(
            start[0] + (end[0] - start[0]) * step / steps,
            start[1] + (end[1] - start[1]) * step / steps,
        )
    page.mouse.up()


def _enter_edit_mode(page: Page) -> None:
    """Open edit mode the way an operator does: right-click a bar, pick the row.

    The right-click is the entry point with no prerequisites (the display-menu
    row needs the popover open, the palette action needs the palette open), so
    it is the one every drag test uses to get to the sheet.
    """
    host = page.locator(HEADER_HOST)
    host.click(button="right", position={"x": 4, "y": 4})
    page.locator(f'{CONTEXT_MENU} [data-bar-action="customize"]').click()
    expect(page.locator(SHEET)).to_have_class(re.compile(r"\bis-open\b"), timeout=5_000)


def _tile(page: Page, type_: str) -> Locator:
    return page.locator(f'{SHEET} .bar-tile[data-bar-tile="{type_}"]')


def _shell(page: Page, host: str, type_: str) -> Locator:
    return page.locator(f'[data-bar-host="{host}"] > .bar-item[data-bar-item="{type_}"]')


def _outside_the_bars(page: Page) -> tuple[float, float]:
    """A point over neither bar — the release that means "remove"."""
    size = page.viewport_size
    assert size is not None
    return size["width"] / 2, size["height"] / 2


# ---------------------------------------------------------------------------
# (a) a tile dragged in from the sheet
# ---------------------------------------------------------------------------


def test_a_tile_dragged_from_the_sheet_lands_where_it_was_dropped(tmp_path, chromium_browser):
    """The sheet's tiles are drag sources, and the drop point picks the index.

    Dropped on the LEFT half of the first item in the status bar, so the
    assertion is about the position the pointer chose rather than about an
    append that would look the same wherever it landed — and about the item
    arriving ONCE, in the bar the pointer was over.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "space", "display"], status=["clock", "docs"])
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        tile = _tile(page, "stopwatch")
        expect(tile).to_be_enabled()
        clock_box = _shell(page, "status", "clock").bounding_box()
        assert clock_box is not None
        _drag(
            page,
            _center(tile),
            (clock_box["x"] + 2, clock_box["y"] + clock_box["height"] / 2),
        )

        # One gesture is one edit, so the document settles after a single round
        # trip. The wait is a fixed settle rather than a locator retry because
        # the failure this pins is a SECOND write landing behind the first —
        # a polling assertion would see the correct intermediate state and pass.
        page.wait_for_timeout(1_500)
        assert _types(page, "status") == ["stopwatch", "clock", "docs"]
        assert "stopwatch" not in _types(page, "header"), "the tile was added twice"

        page.close()


# ---------------------------------------------------------------------------
# (b) reorder inside one bar
# ---------------------------------------------------------------------------


def test_a_drag_inside_one_bar_reorders_it(tmp_path, chromium_browser):
    """A reorder is a permutation, and a full bar must still accept one.

    The seeded status bar holds three fixed-width items (no ``space``, which
    stretches and would make the midpoint arithmetic depend on the viewport).
    The rightmost is dragged onto the left half of the leftmost.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "space", "display"],
            status=["clock", "stopwatch", "feedback"],
        )
        page = _open(chromium_browser, base_url)
        assert _types(page, "status") == ["clock", "stopwatch", "feedback"]
        _enter_edit_mode(page)

        feedback = _shell(page, "status", "feedback")
        clock_box = _shell(page, "status", "clock").bounding_box()
        assert clock_box is not None
        _drag(
            page,
            _center(feedback),
            (clock_box["x"] + 2, clock_box["y"] + clock_box["height"] / 2),
        )

        expect(page.locator(f"{STATUS_HOST} > .bar-item")).to_have_count(3)
        page.wait_for_function(
            "() => document.querySelector('[data-bar-host=\"status\"] > .bar-item')"
            "?.dataset.barItem === 'feedback'",
            timeout=5_000,
        )
        assert _types(page, "status") == ["feedback", "clock", "stopwatch"]

        page.close()


# ---------------------------------------------------------------------------
# (c) a move across the two bars
# ---------------------------------------------------------------------------


def test_an_item_dragged_to_the_other_bar_leaves_the_first(tmp_path, chromium_browser):
    """A cross-bar move is one edit: the item arrives, and it is gone from home."""
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "clock", "space", "display"],
            status=["docs"],
        )
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        status_box = page.locator(STATUS_HOST).bounding_box()
        assert status_box is not None
        # The far right of the footer: past every shell's midpoint, so the
        # insertion index is the end of the run.
        _drag(
            page,
            _center(_shell(page, "header", "clock")),
            (
                status_box["x"] + status_box["width"] - 2,
                status_box["y"] + status_box["height"] / 2,
            ),
        )

        expect(_shell(page, "status", "clock")).to_be_attached(timeout=5_000)
        assert _types(page, "status") == ["docs", "clock"]
        assert _types(page, "header") == ["logo", "space", "display"]

        page.close()


def test_an_item_dropped_into_the_header_is_not_folded_away(tmp_path, chromium_browser):
    """An item dropped into a bar stays in that bar.

    The mirror of the test above, and the direction that exposes the crowding
    probe: the header ends in an item that never folds, the status bar does
    not. Asserted
    twice over — the bar must not report crowding it does not have, and the
    dropped item must be on screen rather than parked in the pool behind the
    overflow trigger.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "space", "display"], status=["clock", "docs"])
        page = _open(chromium_browser, base_url)

        crowding = (
            "() => {const bar = document.querySelector('[data-bar-host=\"header\"]');"
            " return bar.scrollWidth - bar.clientWidth;}"
        )
        assert page.evaluate(crowding) <= 0, "the header is crowded before editing even starts"
        _enter_edit_mode(page)
        assert page.evaluate(crowding) <= 0, (
            "edit mode alone made the header report crowding; the ladder will fold on the "
            "next reconcile"
        )

        header_box = page.locator(HEADER_HOST).bounding_box()
        assert header_box is not None
        _drag(
            page,
            _center(_shell(page, "status", "clock")),
            (
                header_box["x"] + header_box["width"] - 2,
                header_box["y"] + header_box["height"] / 2,
            ),
        )

        expect(_shell(page, "header", "clock")).to_be_attached(timeout=5_000)
        assert _types(page, "header") == ["logo", "space", "display", "clock"]
        assert _types(page, "status") == ["docs"]

        page.close()


# ---------------------------------------------------------------------------
# (d) released over neither bar
# ---------------------------------------------------------------------------


def test_an_item_released_outside_both_bars_is_removed(tmp_path, chromium_browser):
    """Only a release over NEITHER bar removes — the drag's one destructive end."""
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "space", "display"], status=["clock", "docs"])
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        _drag(page, _center(_shell(page, "status", "docs")), _outside_the_bars(page))

        expect(_shell(page, "status", "docs")).to_have_count(0, timeout=5_000)
        assert _types(page, "status") == ["clock"]

        page.close()


# ---------------------------------------------------------------------------
# (e) the arrangement survives a reload
# ---------------------------------------------------------------------------


def test_the_arrangement_survives_a_reload(tmp_path, chromium_browser, store_dir):
    """The layout is stored server-side, so the SECOND first-paint carries it.

    Proved from both ends: the reloaded page renders the edited order, and the
    document is on disk in this test's throwaway store rather than in the tab.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "space", "display"], status=["clock", "docs"])
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        _drag(page, _center(_shell(page, "status", "docs")), _outside_the_bars(page))
        expect(_shell(page, "status", "docs")).to_have_count(0, timeout=5_000)

        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(HYDRATED_SHELL, timeout=15_000)
        assert _types(page, "status") == ["clock"]

        stored = store_dir / LAYOUT_FILENAME
        assert stored.is_file(), f"no layout document under {store_dir}"

        page.close()


# ---------------------------------------------------------------------------
# (f) no item is exempt from removal
# ---------------------------------------------------------------------------


def test_the_wordmark_dropped_outside_is_removed_like_any_item(tmp_path, chromium_browser):
    """Nothing in the bars is locked: the wordmark goes the way the clock does.

    The terminal depends on no bar item being present — the command palette
    still reaches Customize, Settings, the mode switch and Log out without the
    header items — so there is no item the operator may not remove.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "space", "display"], status=["clock"])
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        _drag(page, _center(_shell(page, "header", "logo")), _outside_the_bars(page))

        expect(_shell(page, "header", "logo")).to_have_count(0, timeout=5_000)
        assert _types(page, "header") == ["space", "display"]

        page.close()


# ---------------------------------------------------------------------------
# (g) refusals: by host, and by deployment
# ---------------------------------------------------------------------------


def test_a_header_item_moves_to_the_status_bar(tmp_path, chromium_browser):
    """One move and one refusal, the refusal stated rather than silent.

    Dragging the search trigger down onto the footer moves it there and it
    keeps working: every type may sit in either bar (this drop used to be
    refused as "Header only"). A type this deployment cannot render is a
    DISABLED tile wearing the reason, so the operator reads it without a
    gesture at all.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(base_url, header=["logo", "search", "space", "display"], status=["clock"])
        page = _open(chromium_browser, base_url)
        _enter_edit_mode(page)

        status_box = page.locator(STATUS_HOST).bounding_box()
        assert status_box is not None
        _drag(
            page,
            _center(_shell(page, "header", "search")),
            (status_box["x"] + status_box["width"] / 2, status_box["y"] + status_box["height"] / 2),
        )

        expect(_shell(page, "status", "search")).to_be_visible(timeout=5_000)
        assert "search" in _types(page, "status")
        assert _types(page, "header") == ["logo", "space", "display"]
        # The trigger is the same node, at the status bar's density.
        trigger = page.locator(f"{STATUS_HOST} #command-palette-btn")
        expect(trigger).to_be_visible()
        box = trigger.bounding_box()
        assert box is not None and box["height"] <= 22, box

        # system-health needs the SYSTEM panel; this deployment enables only
        # `artifacts`.
        unavailable = _tile(page, "system-health")
        expect(unavailable).to_be_disabled()
        expect(unavailable.locator(".bar-tile-reason")).to_have_text("Not in this deployment")

        page.close()


# ---------------------------------------------------------------------------
# (h) Simple mode
# ---------------------------------------------------------------------------


def test_simple_mode_renders_the_layout_and_offers_no_way_to_change_it(tmp_path, chromium_browser):
    """The mode axis is not the layout axis.

    Simple renders the same saved arrangement — the bars are the operator's
    chrome, not the mode's — and withholds only the three ways in. Absent, not
    disabled: a control that is visible and does nothing is worse than none.

    Each of the three absences is paired with its presence in Expert, on a cold
    page of the same server. An absence on its own is not evidence: an entry
    point that fails to mount in BOTH modes reads identically here.
    """
    with _launch_web_terminal(tmp_path) as (base_url, app):
        _seed_layout(
            base_url,
            header=["logo", "space", "search", "display"],
            status=["clock", "docs"],
        )
        app.state.web_ui_mode = "simple"
        page = _open(chromium_browser, base_url)

        expect(page.locator("html")).to_have_attribute("data-ui-mode", "simple")
        assert _types(page, "header") == ["logo", "space", "search", "display"]
        assert _types(page, "status") == ["clock", "docs"]

        # 1. right-click either bar: no menu, and nothing was consumed.
        page.locator(HEADER_HOST).click(button="right", position={"x": 4, "y": 4})
        expect(page.locator(CONTEXT_MENU)).to_have_count(0)
        expect(page.locator(SHEET)).to_have_count(0)

        # 2. the display menu's projected row.
        page.locator(DISPLAY_MENU_BTN).click()
        expect(page.locator(DISPLAY_MENU_CARD)).to_have_class(
            re.compile(r"\bopen\b"), timeout=2_000
        )
        expect(page.locator(CUSTOMIZE_ROW)).to_have_count(0)
        page.keyboard.press("Escape")

        # 3. the palette action.
        page.locator(PALETTE_BTN).click()
        palette_input = page.locator(PALETTE_INPUT)
        expect(palette_input).to_be_visible(timeout=5_000)
        palette_input.fill("Customize")
        expect(page.locator(PALETTE_ITEM).filter(has_text="Customize bars")).to_have_count(0)
        page.close()

        # The same three, in Expert, on a cold page. Without this the three
        # absences above would pass just as well against entry points that never
        # mount at all -- which is what the display-menu row used to do, before
        # it learned to wait for `osprey-display-menu` to be defined.
        app.state.web_ui_mode = "expert"
        expert = _open(chromium_browser, base_url)
        assert _types(expert, "header") == ["logo", "space", "search", "display"]

        expert.locator(HEADER_HOST).click(button="right", position={"x": 4, "y": 4})
        expect(expert.locator(f'{CONTEXT_MENU} [data-bar-action="customize"]')).to_be_visible()
        expert.keyboard.press("Escape")

        expert.locator(DISPLAY_MENU_BTN).click()
        expect(expert.locator(DISPLAY_MENU_CARD)).to_have_class(
            re.compile(r"\bopen\b"), timeout=2_000
        )
        expect(expert.locator(CUSTOMIZE_ROW)).to_have_count(1)
        expert.keyboard.press("Escape")

        expert.locator(PALETTE_BTN).click()
        expert_palette = expert.locator(PALETTE_INPUT)
        expect(expert_palette).to_be_visible(timeout=5_000)
        expert_palette.fill("Customize")
        expect(expert.locator(PALETTE_ITEM).filter(has_text="Customize bars")).to_have_count(1)

        expert.close()


# ---------------------------------------------------------------------------
# (i) a header packed to the cap does not overflow
# ---------------------------------------------------------------------------


def test_a_header_full_of_max_width_gaps_does_not_overflow_at_1024(tmp_path, chromium_browser):
    """The declared flex hints, not a JS rung, are what hold the bar together.

    Fifteen gaps at their maximum 400 px ask for 6000 px inside a 1024 px
    viewport. Each declares ``flex: 0 1 <size>px`` with ``min-width: 0``, so
    spacing YIELDS continuously and the chrome beside it is never
    clipped. The bar is filled to the per-host cap, which is the worst case the
    store will accept.

    Asserted as a MARGIN, not just as a fit — see :data:`MIN_BAR_HEADROOM_RATIO`.
    """
    gaps: list[str | dict] = [{"type": "space", "options": {"width": 400}} for _ in range(15)]
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "identity", "control-target", "search", "display", *gaps],
            status=["clock"],
        )
        page = _open(chromium_browser, base_url, viewport=NARROW_VIEWPORT)

        overflow = page.evaluate(
            "() => {"
            " const root = document.documentElement;"
            " const bar = document.querySelector('[data-bar-host=\"header\"]');"
            " return {"
            "  page: root.scrollWidth - root.clientWidth,"
            "  bar: bar.scrollWidth - bar.clientWidth,"
            "  body: document.body.scrollWidth - document.body.clientWidth,"
            " };"
            "}"
        )
        # All three are checked at the same zero tolerance, and all three are
        # checked because they can disagree: a run that fits its own host still
        # widens the page if something inside it escapes the host's box, and the
        # page/body pair is what catches that. (An earlier draft allowed the bar
        # 1px of slack; it never needed it, and slack in the one assertion that
        # is about arithmetic hides the arithmetic going wrong.)
        assert overflow["page"] <= 0, f"the page scrolls horizontally: {overflow}"
        assert overflow["body"] <= 0, f"the body scrolls horizontally: {overflow}"
        assert overflow["bar"] <= 0, f"the header run scrolls horizontally: {overflow}"

        # "It fits" is worth little without "by how much". The bar's
        # NON-SHRINKING content is every item that is not a gap, plus the
        # column-gap between all of them; the gaps are the only thing that can
        # yield, so that figure is the floor the run can never go below. This
        # runner is macOS; Linux headless chromium renders the same system font
        # stack wider, and `search` is the widest text-bearing item here, so a
        # bare pass on this machine says nothing about CI. A required margin
        # does: at 1024px the floor measures ~405px of a 996px bar, so a quarter
        # of the bar is a wide bound that still fails loudly if a gap stops
        # shrinking or the chrome doubles.
        floor = page.evaluate(
            "() => {"
            " const bar = document.querySelector('[data-bar-host=\"header\"]');"
            " const kids = [...bar.children];"
            " const gap = parseFloat(getComputedStyle(bar).columnGap) || 0;"
            " const fixed = kids"
            "  .filter((kid) => kid.dataset.barItem !== 'space')"
            "  .reduce((sum, kid) => sum + kid.offsetWidth, 0);"
            " return {"
            "  fixed: Math.round(fixed + gap * Math.max(0, kids.length - 1)),"
            "  available: bar.clientWidth,"
            " };"
            "}"
        )
        headroom = floor["available"] - floor["fixed"]
        assert headroom >= MIN_BAR_HEADROOM_RATIO * floor["available"], (
            f"the header's non-shrinking content leaves only {headroom}px of "
            f"{floor['available']}px; a wider font stack on another runner would "
            f"clip it: {floor}"
        )

        # The header chrome is still on screen and reachable, not squeezed past
        # the right edge -- the reason gaps declare a shrink at all.
        for selector in (PALETTE_BTN, DISPLAY_MENU_BTN):
            control = page.locator(selector)
            expect(control).to_be_visible()
            box = control.bounding_box()
            assert box is not None
            assert box["x"] >= 0 and box["x"] + box["width"] <= NARROW_VIEWPORT["width"] + 1, (
                f"{selector} is outside the 1024px viewport: {box}"
            )

        page.close()


# ---------------------------------------------------------------------------
# (j) the status readouts are live
# ---------------------------------------------------------------------------


def test_the_status_readouts_are_live_on_a_real_page(tmp_path, chromium_browser):
    """The clock reports the page it is on.

    The unit suites drive the builder by hand; this is the one place it is
    rendered against a live terminal, through the real boot path.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "space", "display"],
            status=["clock"],
        )
        page = _open(chromium_browser, base_url)

        expect(page.locator(f"{STATUS_HOST} .bar-clock-time")).to_have_text(
            re.compile(r"^\d{2}:\d{2}$"), timeout=10_000
        )

        page.close()


# ---------------------------------------------------------------------------
# (k) a plain deployment paints no separator
# ---------------------------------------------------------------------------


def test_a_deployment_with_no_identity_block_paints_no_separator(tmp_path, chromium_browser):
    """The middot is a fact about ORDER and about the identity item, both.

    With nothing to identify — no signed-in user and no deployment name — the
    identity shell is ABSENT and the spacer inherits the logo's follower slot.
    The separator rule names the identity item as well as its position exactly
    so that spacer stays bare; without that, every single-user deployment would
    paint a dangling middot after the wordmark.
    """
    with _launch_web_terminal(tmp_path) as (base_url, app):
        _seed_layout(
            base_url,
            header=["logo", "identity", "space", "display"],
            status=["clock"],
        )
        app.state.app_name = ""
        app.state.terminal_user = ""
        page = _open(chromium_browser, base_url)

        assert _types(page, "header") == ["logo", "space", "display"]
        follower = page.locator(f'{HEADER_HOST} > .bar-item[data-follows="logo"]')
        expect(follower).to_have_attribute("data-bar-item", "space")
        content = page.evaluate(
            "() => getComputedStyle("
            ' document.querySelector(\'[data-bar-host="header"] >'
            " .bar-item[data-follows=\"logo\"]'), '::before').content"
        )
        assert content in ("none", "normal", '""'), f"a separator was painted: {content!r}"
        assert "·" not in (page.locator(HEADER_HOST).inner_text() or "")

        page.close()


# ---------------------------------------------------------------------------
# (l) reset to default
# ---------------------------------------------------------------------------


def test_reset_to_default_discards_the_stored_arrangement(tmp_path, chromium_browser, store_dir):
    """A reset is a DELETE, not a write of the deployment's own layout.

    "Saved nothing" and "saved something equal to the default" behave
    differently from here on, so the assertion is that the document leaves the
    store entirely and the bars come back as this deployment renders them.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "space", "display"],
            status=["stopwatch", "clock"],
        )
        page = _open(chromium_browser, base_url)
        assert _types(page, "status") == ["stopwatch", "clock"]
        assert (store_dir / LAYOUT_FILENAME).is_file()

        # The Default preset is the sheet's one pill, so it exists only while
        # editing.
        _enter_edit_mode(page)
        page.locator('.bar-sheet-presets [data-bar-preset="default"]').click()

        # The deployment default: a space, then the clock at the right edge.
        # `system-health` is dropped because the SYSTEM panel is not enabled
        # here.
        page.wait_for_function(
            "() => !document.querySelector("
            ' \'[data-bar-host="status"] > .bar-item[data-bar-item="stopwatch"]\')',
            timeout=5_000,
        )
        assert _types(page, "status") == ["space", "clock"]
        assert not (store_dir / LAYOUT_FILENAME).exists(), (
            "a reset must remove the document, not rewrite it with the default"
        )

        page.close()


# ---------------------------------------------------------------------------
# (m) a hidden header comes back from a tile header's menu
# ---------------------------------------------------------------------------

#: The terminal tile's menu is wired on the adopted `.terminal-header` node,
#: not on the tab around it, so the press lands on the session label -- the
#: same way the panels suite reaches a service tile's menu through its title.
TERMINAL_TILE_HEADER = ".tile-tab-terminal .terminal-label"
PANEL_MENU = ".rail-context-menu"
PANEL_MENU_ITEM = ".rail-context-item"


def test_a_hidden_header_comes_back_from_the_terminal_tile_menu(tmp_path, chromium_browser):
    """Hiding the header must not strand the operator.

    The header's own right-click menu is what hides it, and a hidden bar has
    no surface left to right-click. The tile headers are what remains on
    screen, so every one of them offers **Show header** while the header is
    hidden -- proved here on the terminal tile, from a page that loaded with
    the header already hidden (the server-stamped state, not a client toggle).
    The restore is a stored change: a reload comes back with the header shown.
    """
    with _launch_web_terminal(tmp_path) as (base_url, _app):
        _seed_layout(
            base_url,
            header=["logo", "space", "display"],
            status=["clock", "docs"],
            header_visible=False,
        )
        page = chromium_browser.new_page(viewport=VIEWPORT)
        page.add_init_script(_DISMISS_TOUR)
        page.goto(base_url, wait_until="domcontentloaded")
        # The header is withdrawn, so its hydrated shell is attached but not
        # visible -- the default "visible" wait would sit out the timeout.
        page.wait_for_selector(HYDRATED_SHELL, state="attached", timeout=15_000)
        expect(page.locator("html")).to_have_attribute("data-header-bar", "hidden")
        expect(page.locator(HEADER_HOST)).to_be_hidden()

        page.locator(TERMINAL_TILE_HEADER).click(button="right")
        menu = page.locator(PANEL_MENU)
        expect(menu).to_have_count(1, timeout=5_000)
        show = menu.locator(PANEL_MENU_ITEM).filter(has_text="Show header")
        expect(show).to_have_count(1)
        show.click()

        expect(page.locator(HEADER_HOST)).to_be_visible(timeout=5_000)
        expect(page.locator("html")).not_to_have_attribute("data-header-bar", "hidden")

        # With the header back, the tile menu carries only the terminal's verbs.
        page.locator(TERMINAL_TILE_HEADER).click(button="right")
        expect(page.locator(PANEL_MENU)).to_have_count(1, timeout=5_000)
        expect(page.locator(PANEL_MENU_ITEM).filter(has_text="Show header")).to_have_count(0)
        page.keyboard.press("Escape")

        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(HYDRATED_SHELL, timeout=15_000)
        expect(page.locator("html")).not_to_have_attribute("data-header-bar", "hidden")

        page.close()
