"""Browser tests: the Workspace divider and the session tree in a real layout engine.

The unit suite (``sidebar-divider.test.mjs``) pins the cap arithmetic, the
fraction storage, the snap and the keys against happy-dom, which lays nothing
out. What only a browser can show is how the group sizes, and that its rows
keep their height when it runs out of room.

Coverage:

  (1) with nothing stored, the group is as tall as its content (an empty
      project keeps the line high) and grows by a row per session.
  (2) ~30 sessions at an 800 px window: the group stops at the cap and
      scrolls; every session row stays 30 px, the open session's artifact
      box keeps five and a half rows, and nothing spills out of the tree.
  (3) a drag is stored as a fraction of the rail, which holds when the window
      grows; a drag back within 10 px of the automatic height snaps and clears the stored height.
  (4) a whole-pixel height stored by an earlier build keeps its height and is
      rewritten once as a fraction.
  (5) at about 800 px a drag down past the room stops where every group still
      fits: the line tracks the pointer, then stays on the group's end, and
      neither the sidebar column nor the rail scrolls under the Search box.
  (6) a stored share too tall for the window is clamped on load and on resize.

The sessions of (1) and (2) are built with the tree's own row builders
(``sidebar-artifact-rows.js``) into the live tree of an empty project. The
tree draws itself from the hub's reads (the session listing, the artifacts
panel turning served), which land at their own pace and later on a loaded
machine; so the seed stops the tree first and keeps it from drawing over the
seeded rows.

Every case starts once the sidebar has its last row: the Docs row in the foot
appears when its own read of ``/api/panels`` lands, after the divider is wired,
and the rail's room (so the cap, a stored share, an earlier build's rewrite)
shrinks with it.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_divider_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from tests.interfaces._hub import hub_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_KEY = "osprey-sidebar-workspace-height"
_GROUP = "#panel-rail > .sidebar-group:has(#sidebar-agent-row)"
_DIVIDER = "#panel-rail > .sidebar-divider"

#: Stops the tree's own renderer, fills the live tree with ``past`` closed
#: sessions under an open current session holding ``arts`` artifact rows, then
#: reports the layout one frame on.
_SEED_JS = """async ([past, arts]) => {
  const rows = await import('/static/js/sidebar-artifact-rows.js');
  // The page's own module instance (same URL). The reset stops the tree's reads
  // and listeners; the artifacts panel turning served later restarts the tree,
  // so its draws (all through replaceChildren) are also kept off the seed.
  const artifacts = await import('/static/js/sidebar-artifacts.js');
  artifacts.resetSidebarArtifacts();
  const tree = document.getElementById('sidebar-session-tree');
  const draw = (...nodes) => Element.prototype.replaceChildren.apply(tree, nodes);
  tree.replaceChildren = () => {};
  const session = (i, state) => ({ transcript_id: `t${i}`, title: `Session ${i}`, state });
  const kids = [];
  for (let i = 0; i < arts; i++) {
    const row = document.createElement('button');
    row.className = 'osprey-list-row osprey-tree-row osprey-tree-d2';
    row.textContent = `Artifact ${i}`;
    kids.push(row);
  }
  const nodes = [
    rows.sessionRow(session(0, 'live'), { active: true, open: true, count: arts }),
    rows.sessionBox('t0', kids),
  ];
  for (let i = 1; i <= past; i++)
    nodes.push(rows.sessionRow(session(i, 'idle'), { active: false, open: false, count: 2 }));
  draw(...nodes);
  window.__seededFirst = nodes[0];
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return window.__dividerLayout();
}"""

#: The measurements every case reads.
_LAYOUT_JS = """() => {
  window.__dividerLayout = () => {
    const group = document.querySelector('#panel-rail > .sidebar-group:has(#sidebar-agent-row)');
    const tree = document.getElementById('sidebar-session-tree');
    const box = tree.querySelector('.osprey-tree-box');
    const tools = document.querySelector('#panel-rail > [data-group="tools"]');
    return {
      height: group.offsetHeight,
      content: group.scrollHeight,
      scrolls: group.scrollHeight - group.clientHeight > 2,
      sized: group.hasAttribute('data-sized'),
      cap: parseFloat(group.style.getPropertyValue('--sidebar-group-cap')),
      rowHeights: [...tree.querySelectorAll('.osprey-tree-session')].map((r) => r.offsetHeight),
      box: box ? { height: box.offsetHeight, full: box.scrollHeight } : null,
      treeSpill: tree.scrollHeight - tree.clientHeight,
      gap: tools.getBoundingClientRect().top - group.getBoundingClientRect().bottom,
      stored: localStorage.getItem('osprey-sidebar-workspace-height'),
      seeded: Boolean(window.__seededFirst) && tree.firstElementChild === window.__seededFirst,
    };
  };
}"""


@contextmanager
def _hub(workspace_dir: Path) -> Iterator[str]:
    """A real web-terminal hub over an empty project; yields its base URL."""
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace_dir
    )
    with hub_server(workspace_dir, shared_root) as (base_url, _app):
        yield base_url


@contextmanager
def _page(browser: Browser, url: str, height: int, stored: str | None = None) -> Iterator[Page]:
    """Open the hub at 1280×``height``, ``stored`` preset under the height key."""
    ctx = browser.new_context(viewport={"width": 1280, "height": height})
    if stored is not None:
        ctx.add_init_script(
            f"if (!sessionStorage.getItem('seeded')) {{"
            f" localStorage.setItem('{_KEY}', '{stored}'); sessionStorage.setItem('seeded', '1'); }}"
        )
    page = ctx.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded")
        _ready(page)
        yield page
    finally:
        ctx.close()


def _ready(page: Page) -> None:
    """Wait for the divider and for the sidebar's last row, then wire the measurements.

    The divider repaints from a ResizeObserver on the sidebar's blocks, which
    runs in the frame the Docs row first lays out, before any later evaluate.
    """
    expect(page.locator(f"{_DIVIDER}[aria-valuetext]")).to_be_attached(timeout=10_000)
    expect(page.locator("#panel-docs-link")).to_be_visible(timeout=10_000)
    page.evaluate(_LAYOUT_JS)


def _layout(page: Page) -> dict[str, Any]:
    return page.evaluate("window.__dividerLayout()")


def _seed(page: Page, past: int, arts: int) -> dict[str, Any]:
    return page.evaluate(_SEED_JS, [past, arts])


def _divider_y(page: Page) -> tuple[float, float]:
    box = page.locator(_DIVIDER).bounding_box()
    assert box is not None, "the divider has no layout box"
    return box["x"] + box["width"] / 2, box["y"]


#: The room the sidebar leaves the rail (sidebar-divider.js railRoom).
_RAIL_ROOM_JS = """() => {
  const side = document.getElementById('panel-rail').parentElement;
  const cs = getComputedStyle(side);
  let room = side.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  for (const el of side.children) {
    if (el.id === 'panel-rail') continue;
    const ecs = getComputedStyle(el);
    if (ecs.position !== 'absolute' && ecs.position !== 'fixed')
      room -= el.offsetHeight + parseFloat(ecs.marginTop) + parseFloat(ecs.marginBottom);
  }
  return room;
}"""


def _rail_room(page: Page) -> float:
    """The room the sidebar leaves the rail (sidebar-divider.js railRoom)."""
    return page.evaluate(_RAIL_ROOM_JS)


def test_automatic_height_follows_a_small_project(tmp_path, chromium_browser):
    """Empty: the group hugs its content, line high; four sessions add four rows."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, 800) as page:
        empty = _seed(page, 0, 0)
        assert not empty["sized"]
        assert not empty["scrolls"]
        assert empty["height"] < 200, empty
        assert empty["cap"] >= 240, empty

        four = _seed(page, 4, 0)
        assert not four["scrolls"]
        assert four["height"] == empty["height"] + 4 * 30, (empty, four)
        assert four["stored"] is None


def test_thirty_sessions_scroll_in_the_group_and_no_row_squashes(tmp_path, chromium_browser):
    """At 800 px with ~30 sessions the group stops at the cap; rows stay 30 px."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, 800) as page:
        lay = _seed(page, 30, 6)
        assert lay["seeded"], lay
        assert len(lay["rowHeights"]) == 31, lay["rowHeights"]
        assert set(lay["rowHeights"]) == {30}, lay["rowHeights"]
        # The open session's box: min(its content, 5½ rows of 26 px + 2 px padding).
        assert lay["box"]["height"] >= min(lay["box"]["full"], 145), lay["box"]
        assert 240 <= lay["cap"] <= 480, lay
        assert lay["height"] == lay["cap"], lay
        assert lay["scrolls"], lay
        # Nothing hangs out of the tree, and Tools starts below the group.
        assert lay["treeSpill"] <= 1, lay
        assert lay["gap"] >= -1, lay

        # The same holds in a dragged group shorter than the tree.
        x, y = _divider_y(page)
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y - 100, steps=4)
        page.mouse.up()
        dragged = _layout(page)
        assert dragged["seeded"], dragged
        assert dragged["sized"], dragged
        assert set(dragged["rowHeights"]) == {30}, dragged["rowHeights"]
        assert dragged["box"]["height"] >= min(dragged["box"]["full"], 145), dragged["box"]
        assert dragged["treeSpill"] <= 1, dragged


def test_drag_is_kept_as_a_fraction_and_snaps_back_to_automatic(tmp_path, chromium_browser):
    """A drag keeps its share of the rail at 1100 px; a drag back to the automatic height resets."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, 800) as page:
        auto = _seed(page, 4, 0)
        x, y = _divider_y(page)
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 80, steps=4)
        assert page.locator(f"{_DIVIDER} .sidebar-divider-default").count() == 0
        assert page.locator(_DIVIDER).get_attribute("data-tip") is None
        assert page.locator(_DIVIDER).get_attribute("data-tip-detail") is None
        page.mouse.up()

        dragged = _layout(page)
        assert dragged["sized"]
        assert dragged["height"] == auto["height"] + 80, (auto, dragged)
        assert re.fullmatch(r"0\.\d{4}", dragged["stored"]), dragged["stored"]
        fraction = float(dragged["stored"])
        assert abs(fraction * _rail_room(page) - dragged["height"]) <= 1

        page.set_viewport_size({"width": 1280, "height": 1100})
        page.wait_for_function(
            f"document.querySelector('{_GROUP}').offsetHeight > {dragged['height']}"
        )
        taller = _layout(page)
        assert abs(taller["height"] - round(fraction * _rail_room(page))) <= 1, taller
        assert taller["stored"] == dragged["stored"]

        # Back to within 10 px of the automatic height: snapped, and the stored height cleared.
        x, y = _divider_y(page)
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y - (taller["height"] - auto["height"]) + 6, steps=4)
        page.mouse.up()
        back = _layout(page)
        assert not back["sized"], back
        assert back["stored"] is None
        assert back["height"] == auto["height"], (auto, back)


def test_an_earlier_builds_pixels_become_a_fraction_once(tmp_path, chromium_browser):
    """A stored ``300`` keeps 300 px and is rewritten as its fraction of the rail."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, 800, stored="300") as page:
        lay = _layout(page)
        assert lay["sized"]
        assert lay["height"] == 300, lay
        assert re.fullmatch(r"0\.\d{4}", lay["stored"]), lay["stored"]
        # The rail's room shrinks as the late blocks render and the fraction is
        # rewritten after it, so the two agree once the sidebar's head is done.
        page.wait_for_function(
            f"Math.abs(Number(localStorage.getItem('{_KEY}')) * ({_RAIL_ROOM_JS})() - 300) <= 1",
            timeout=10_000,
        )

        # The fraction is rewritten until the sidebar's late blocks (the control
        # target, the footer) have rendered, up to the moment this page goes, so
        # the one the next load reads is the one stored as the page is hidden.
        # The tab's sessionStorage carries it across the reload.
        page.evaluate(
            "addEventListener('pagehide', () => sessionStorage.setItem("
            f"'written', localStorage.getItem('{_KEY}')))"
        )
        page.reload(wait_until="domcontentloaded")
        _ready(page)
        # The fraction is applied to the rail's room as it stands, which the
        # late blocks shrink as they render: the height is 300 px once they have.
        page.wait_for_function("window.__dividerLayout().height === 300", timeout=10_000)
        again = _layout(page)
        assert again["height"] == 300, again
        assert again["stored"] == page.evaluate("sessionStorage.getItem('written')")


#: Where the short-window cases read their positions: the divider, the Search
#: box, the group's top, how far the sidebar column and the rail scrolled, and
#: the rail's room left below its last group.
_PLACES_JS = """() => {
  const rail = document.getElementById('panel-rail');
  const group = rail.querySelector(':scope > .sidebar-group:has(#sidebar-agent-row)');
  const search = document.querySelector('#sidebar .osprey-search');
  return {
    divider: rail.querySelector(':scope > .sidebar-divider').getBoundingClientRect().top,
    searchBottom: search.getBoundingClientRect().bottom,
    groupTop: group.getBoundingClientRect().top,
    height: group.offsetHeight,
    sidebarScroll: rail.parentElement.scrollTop,
    railScroll: rail.scrollTop,
    railSpill: rail.scrollHeight - rail.clientHeight,
    railSlack: rail.getBoundingClientRect().bottom - rail.lastElementChild.getBoundingClientRect().bottom,
    stored: localStorage.getItem('osprey-sidebar-workspace-height'),
  };
}"""


@pytest.mark.parametrize("height", [800, 813])
def test_a_drag_past_the_room_stops_where_everything_fits(tmp_path, chromium_browser, height):
    """On a short window a drag down stops at the room left; the line stays on the group's end."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, height) as page:
        _seed(page, 4, 0)
        before = page.evaluate(_PLACES_JS)
        x, y = _divider_y(page)
        page.mouse.move(x, y)
        page.mouse.down()
        # Within the room the line is under the pointer.
        page.mouse.move(x, y + 40, steps=4)
        assert abs(page.evaluate(_PLACES_JS)["divider"] - (y + 40)) <= 1
        page.mouse.move(x, y + 500, steps=10)
        during = page.evaluate(_PLACES_JS)
        page.mouse.up()
        after = page.evaluate(_PLACES_JS)

        for lay in (during, after):
            assert lay["sidebarScroll"] == 0, f"{lay}"
            assert lay["railScroll"] == 0, f"{lay}"
            assert lay["railSpill"] <= 1, f"{lay}"
            # The group grew, and its top stayed put below the Search box.
            assert lay["height"] > before["height"], f"{before} {lay}"
            assert lay["groupTop"] >= lay["searchBottom"] - 1, f"{lay}"
            assert abs(lay["groupTop"] - before["groupTop"]) <= 1, f"{before} {lay}"
            # The line moved down by exactly the growth.
            grown = before["divider"] + lay["height"] - before["height"]
            assert abs(lay["divider"] - grown) <= 1, f"{before} {lay}"
            # Clamped at the room: the other groups end on the rail's end.
            assert abs(lay["railSlack"]) <= 1, f"{lay}"
        assert after["height"] == during["height"], (during, after)
        assert after["stored"] is not None


def test_a_height_saved_on_a_tall_window_fits_a_short_one(tmp_path, chromium_browser):
    """A stored share that overflows at 800 px is clamped on load and on resize."""
    with _hub(tmp_path) as url, _page(chromium_browser, url, 800, stored="0.9000") as page:
        _seed(page, 4, 0)
        lay = page.evaluate(_PLACES_JS)
        assert lay["railSpill"] <= 1, f"{lay}"
        assert lay["sidebarScroll"] == 0, f"{lay}"
        assert lay["railScroll"] == 0, f"{lay}"
        assert abs(lay["railSlack"]) <= 1, f"{lay}"
        assert lay["stored"] == "0.9000", f"{lay}"

        page.set_viewport_size({"width": 1280, "height": 700})
        page.wait_for_function(f"document.querySelector('{_GROUP}').offsetHeight < {lay['height']}")
        short = page.evaluate(_PLACES_JS)
        assert short["railSpill"] <= 1, f"{short}"
        assert short["sidebarScroll"] == 0, f"{short}"
        assert short["railScroll"] == 0, f"{short}"
        assert abs(short["railSlack"]) <= 1, f"{short}"
