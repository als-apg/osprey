"""Browser suite: the shared splitter driven live on the artifacts gallery.

``splitter.test.mjs`` proves the module's contract under happy-dom, and
``tests/interfaces/test_shared_splitter_contract.py`` proves every host wires
it. Neither can prove a *real* drag: happy-dom has no layout, and a static
contract cannot see pointer capture, ``requestAnimationFrame`` coalescing, or
whether the geometry a drag writes actually moves anything on screen.

Scope: the artifacts gallery's browse layout
(``interfaces/artifacts/static/js/browse-layout.js``), specifically its
**y-axis** handle ``#resize-handle-y``. The gallery ships two splitters, one
per axis, and parks whichever the current orientation does not use — so the
stacked ("column") layout must be selected before the y handle is live.

Why this host:

- It is the only remaining splitter host with no live browser coverage of its
  own. ``test_gallery_interactions.py`` drives the gallery hard but never
  touches either handle, and ``test_osprey_drawer.py`` covers only the
  drawer's x-axis ``anchor: 'end'`` geometry and its 320px/90vw clamp.
- It is therefore the only place left where a y-axis drag — a different
  delta sign and a different clamp path than the x-axis — is exercised live.

This suite previously drove the event dashboard, which carried three splitters
across two axes. That panel was rebuilt around discrete views and has no
resizable pane left, so the coverage moved here rather than being dropped.

Two assertions from the dashboard era are deliberately NOT ported:

- **Preset / ``storageKey: null`` / ``onCommit`` interplay.** The dashboard was
  the only host that kept one layout record with a preset system and fed the
  splitters through ``onCommit``. No host does that now, so there is nothing
  left to assert; the gallery's handles persist themselves under their own
  storage keys, which ``splitter.test.mjs`` already covers.
- **The computed ``transition-duration`` half of the drag-suppression check.**
  ``base.css`` suppresses transitions via ``[data-splitter-dragging]``, but the
  assertion is only meaningful on a pane that declares a transition on the
  dragged dimension. The dashboard's ``.timeline-zone`` did; ``.browse-sidebar``
  does not, so asserting ``0s`` here would pass no matter what the rule did.
  The attribute half — applied for the drag, dropped on release — IS ported,
  because that is the mechanism itself and it is non-vacuous on any host.

Placement rationale: ``design_system`` already owns the pattern of
browser-proving a design-system module against a live consumer app (see
``test_osprey_drawer.py`` and ``test_behavioral.py``), and the module under
test is ``design_system/static/js/splitter.js``.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_splitter_surfaces.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1280, "height": 900}

#: How long to wait for browse-layout.js's `initSplitter` call to land.
WIRE_TIMEOUT_MS = 10_000

#: browse-layout.js's ORIENT_KEY. Seeding it before the module runs is what
#: puts the layout in stacked mode, which is the orientation that enables the
#: y-axis handle (`isEnabled: () => effectiveOrient() === 'column'`).
ORIENT_KEY = "osprey-artifacts-browse-orient"

#: browse-layout.js's HEIGHT_KEY. Seeding a size does two jobs: it pins a
#: deterministic starting height (between browse-layout.js's 120/480 bounds),
#: and it makes the splitter's restore() path call applySize — which is what
#: stamps `aria-valuenow`. Without a stored size restore() returns early and
#: never stamps, so there would be no signal that the module had wired up.
HEIGHT_KEY = "osprey-artifacts-browse-sidebar-height"
SEEDED_HEIGHT = 240

#: browse-layout.js's KEY_STEP — one arrow press.
KEY_STEP = 16

Y_HANDLE = "#resize-handle-y"
PANE = "#browse-sidebar"


@pytest.fixture
def gallery_url(tmp_path, monkeypatch) -> Iterator[str]:
    """Serve the artifacts gallery, which mounts /design-system/* alongside itself.

    Both from one origin matters: the gallery imports the splitter with a
    root-relative specifier, so it must resolve against the gallery itself.

    No fixture artifacts are seeded — the browse layout and both handles exist
    whether or not the tree has content, and this suite asserts geometry only.
    """
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with _run_app_server(app) as base_url:
        yield base_url


def _open_stacked(page: Page, base_url: str) -> None:
    """Load the gallery in stacked mode with the y-axis splitter wired.

    Both keys must be in localStorage *before* browse-layout.js runs, so this
    loads once to obtain the origin, seeds them, then reloads.

    ``aria-valuenow`` is the signal to wait on: the splitter writes it on every
    geometry application, so its presence means the module resolved AND the
    seeded size has been pushed through.
    """
    page.goto(base_url, wait_until="domcontentloaded")
    page.evaluate(
        "([orientKey, heightKey, height]) => {"
        "  localStorage.setItem(orientKey, 'column');"
        "  localStorage.setItem(heightKey, String(height));"
        "}",
        [ORIENT_KEY, HEIGHT_KEY, SEEDED_HEIGHT],
    )
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function(
        "sel => document.querySelector(sel)?.hasAttribute('aria-valuenow')",
        arg=Y_HANDLE,
        timeout=WIRE_TIMEOUT_MS,
    )


def _height(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no bounding box -- is it rendered?"
    return box["height"]


def _drag(page: Page, handle: str, dx: int = 0, dy: int = 0) -> None:
    """Drag a handle by (dx, dy) with intermediate moves.

    The intermediate steps matter: the splitter coalesces pointermove into
    requestAnimationFrame, so a single jump would not exercise the coalescing
    path at all — and it is the release-ordering around that frame which this
    feature had to get right.
    """
    box = page.locator(handle).bounding_box()
    assert box is not None, f"{handle} has no bounding box -- is it rendered?"
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    for step in (0.34, 0.67, 1.0):
        page.mouse.move(start_x + dx * step, start_y + dy * step)
    page.mouse.up()


# ---------------------------------------------------------------------------
# The divider drags, on the right axis, in the right direction
# ---------------------------------------------------------------------------


def test_y_divider_drags_down_to_grow(chromium_browser: Browser, gallery_url: str) -> None:
    """`axis: 'y'`, `anchor: 'start'` — the pane is above, so down grows it."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_stacked(page, gallery_url)

    before = _height(page, PANE)
    _drag(page, Y_HANDLE, dy=80)
    after = _height(page, PANE)

    assert after == pytest.approx(before + 80, abs=6), (
        f"expected the browse sidebar to grow by ~80px, got {before} -> {after}"
    )
    page.close()


def test_a_drag_stamps_and_clears_the_dragging_attribute(
    chromium_browser: Browser, gallery_url: str
) -> None:
    """The mechanism the drag-transition fix rests on, asserted directly.

    ``base.css`` keys ``transition: none`` off ``[data-splitter-dragging]``. The
    rule is only as good as the attribute: if it is never stamped, a pane that
    transitions its dragged dimension eases toward a target the cursor has
    already left and trails for the whole gesture; if it is never cleared,
    discrete collapses stop animating everywhere.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_stacked(page, gallery_url)

    box = page.locator(Y_HANDLE).bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 + 50)

    pane = page.locator(PANE)
    assert pane.get_attribute("data-splitter-dragging") is not None, (
        "the pane must carry data-splitter-dragging while a drag is in flight"
    )

    page.mouse.up()
    assert pane.get_attribute("data-splitter-dragging") is None, (
        "the attribute must be dropped on release, or discrete collapses stop animating"
    )
    page.close()


# ---------------------------------------------------------------------------
# Collapse restores the size held before collapsing
# ---------------------------------------------------------------------------


def test_double_click_collapses_and_restores_the_prior_height(
    chromium_browser: Browser, gallery_url: str
) -> None:
    """The gesture is only useful if it restores where you left it."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_stacked(page, gallery_url)

    _drag(page, Y_HANDLE, dy=60)
    chosen = _height(page, PANE)

    page.locator(Y_HANDLE).dblclick()
    collapsed = _height(page, PANE)
    assert collapsed < chosen / 2, (
        f"expected the sidebar to collapse well below its {chosen}px height, got {collapsed}"
    )

    page.locator(Y_HANDLE).dblclick()
    restored = _height(page, PANE)
    assert restored == pytest.approx(chosen, abs=4), (
        f"expected the pre-collapse height ~{chosen}px back, got {restored}"
    )
    page.close()


def test_arrow_keys_nudge_on_the_vertical_axis(chromium_browser: Browser, gallery_url: str) -> None:
    """A divider only reachable by pointer is unusable by keyboard."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_stacked(page, gallery_url)

    page.locator(Y_HANDLE).focus()
    before = _height(page, PANE)
    page.keyboard.press("ArrowDown")
    after = _height(page, PANE)

    assert after == pytest.approx(before + KEY_STEP, abs=4), (
        f"ArrowDown should grow the sidebar by one {KEY_STEP}px step, got {before} -> {after}"
    )
    # aria-valuenow is the splitter's own record of what it applied; it must
    # agree with what the box actually measures.
    assert float(page.locator(Y_HANDLE).get_attribute("aria-valuenow")) == pytest.approx(
        after, abs=2
    )
    page.close()
