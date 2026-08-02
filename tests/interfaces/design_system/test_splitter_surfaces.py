"""Browser suite: the shared splitter driven live on the event dashboard.

``splitter.test.mjs`` proves the module's contract under happy-dom, and
``tests/interfaces/test_shared_splitter_contract.py`` proves every host wires
it. Neither can prove a *real* drag: happy-dom has no layout, and a static
contract cannot see pointer capture, ``requestAnimationFrame`` coalescing, or
whether the geometry a drag writes actually moves anything on screen.

Scope: the event dashboard (``src/osprey/dispatch/dashboard.html``), which is
the only splitter host with no browser harness of its own. The other five are
already driven live by suites that predate this one — notably
``tests/interfaces/web_terminal/test_osprey_drawer.py``, which covers the
right-anchored ``anchor: 'end'`` geometry and the 320px/90vw clamp, and
``tests/interfaces/artifacts/test_gallery_interactions.py``.

The dashboard earns its own coverage for three reasons the others do not have:

- it carries the defect this whole feature exists to fix — a ``transition:
  height`` on the pane a drag drives, which made the divider trail the cursor;
- it is the only host with *both* y-axis geometries (``anchor: 'start'`` for
  the timeline above its handle, ``anchor: 'end'`` for the inspector below);
- it is the only host whose splitters do not persist their own state. It keeps
  one layout record plus a preset system, so its splitters take
  ``storageKey: null`` and report through ``onCommit`` — a bridge that a static
  test cannot exercise at all.

It is also, as ``eslint.config.js`` and ``tsconfig.json`` both glob only
``src/osprey/interfaces/**`` JS, the one file in this feature that no linter or
typechecker reads. This suite is the compensating control.

Placement rationale: ``design_system`` already owns the pattern of
browser-proving a design-system module against a live consumer app (see
``test_osprey_drawer.py`` and ``test_behavioral.py``), and the module under
test is ``design_system/static/js/splitter.js``.

Run:
    .venv/bin/pytest tests/interfaces/design_system/test_splitter_surfaces.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect

from tests.interfaces.conftest import _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page

VIEWPORT = {"width": 1280, "height": 900}

#: How long to wait for the dynamic `import()` in wireSplitters() to land.
WIRE_TIMEOUT_MS = 10_000

#: `.timeline-zone` keeps a `transition: height 200ms` for *discrete* state
#: changes — a chevron click, a double-click collapse, a preset. Those must be
#: read after the animation settles.
#:
#: Drags are the opposite case and are read immediately on purpose: the
#: splitter stamps `data-splitter-dragging`, which suppresses that transition
#: for the duration of the gesture. A drag assertion that needed this wait
#: would mean the suppression had stopped working, which is the entire defect
#: this feature exists to prevent.
TRANSITION_SETTLE_MS = 400


@pytest.fixture(autouse=True)
def _reset_mcp_routes() -> Iterator[None]:
    """Reset the shared FastMCP singleton's routes around each test.

    Mirrors ``tests/dispatch/test_design_system_route.py``: ``create_server()``
    mutates a module-level FastMCP singleton and appends dashboard routes to it
    on every call, so each test must start from a clean slate.
    """
    from osprey.dispatch import server

    baseline = list(server.mcp._additional_http_routes)
    yield
    server.mcp._additional_http_routes = baseline


@pytest.fixture
def dashboard_url(tmp_path, monkeypatch) -> Iterator[str]:
    """Serve the dispatcher, which exposes /dashboard and /design-system/* together.

    Both from one origin matters: the dashboard imports the splitter with a
    root-relative specifier, so it resolves against the dispatcher itself
    rather than through the web terminal's panel proxy.
    """
    from osprey.dispatch import server

    # No triggers configured — the splitters need none of that machinery, and
    # create_server() starts with an empty trigger list when the file is absent.
    monkeypatch.setenv("TRIGGERS_YML", str(tmp_path / "does-not-exist.yml"))
    app = server.create_server().http_app()
    with _run_app_server(app) as base_url:
        yield base_url


def _open_dashboard(page: Page, base_url: str) -> None:
    """Navigate to the dashboard and wait for the splitters to be wired.

    ``aria-valuenow`` is the signal: the splitter writes it on every geometry
    application, so its presence means the dynamic import resolved AND the
    stored layout has been pushed through.
    """
    page.goto(f"{base_url}/dashboard", wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelector('#timeline-handle')?.hasAttribute('aria-valuenow')",
        timeout=WIRE_TIMEOUT_MS,
    )


def _height(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no bounding box -- is it rendered?"
    return box["height"]


def _width(page: Page, selector: str) -> float:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no bounding box -- is it rendered?"
    return box["width"]


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
# The three dividers drag, on the right axis, in the right direction
# ---------------------------------------------------------------------------


def test_timeline_divider_drags_down_to_grow(chromium_browser: Browser, dashboard_url: str) -> None:
    """`axis: 'y'`, `anchor: 'start'` — the pane is above, so down grows it."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    before = _height(page, "#timeline-zone")
    _drag(page, "#timeline-handle", dy=80)
    after = _height(page, "#timeline-zone")

    assert after == pytest.approx(before + 80, abs=6), (
        f"expected the timeline zone to grow by ~80px, got {before} -> {after}"
    )
    page.close()


def test_inspector_divider_drags_up_to_grow(chromium_browser: Browser, dashboard_url: str) -> None:
    """`axis: 'y'`, `anchor: 'end'` — the pane is below, so up grows it.

    This is the inverted-delta case; getting the sign wrong makes the divider
    move away from the pointer, which no static check would catch.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    before = _height(page, "#inspector-zone")
    _drag(page, "#inspector-handle", dy=-70)
    after = _height(page, "#inspector-zone")

    assert after == pytest.approx(before + 70, abs=6), (
        f"expected the inspector zone to grow by ~70px dragging up, got {before} -> {after}"
    )
    page.close()


def test_sidebar_divider_drags_right_to_grow(chromium_browser: Browser, dashboard_url: str) -> None:
    """`axis: 'x'`, `anchor: 'start'`, `sizing: 'box'`."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    before = _width(page, "#sidebar")
    _drag(page, "#sidebar-handle", dx=60)
    after = _width(page, "#sidebar")

    assert after == pytest.approx(before + 60, abs=6), (
        f"expected the sidebar to grow by ~60px, got {before} -> {after}"
    )
    page.close()


def test_a_drag_suppresses_the_panes_height_transition(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The defect this feature exists to fix, asserted directly.

    `.timeline-zone` transitions its height. Driving that same property from a
    pointer without suppressing the transition makes the pane ease toward a
    target the cursor has already left, so it trails by the transition duration
    for the whole gesture. The splitter stamps `data-splitter-dragging` for
    exactly as long as the drag lasts, and base.css keys the suppression off it.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    box = page.locator("#timeline-handle").bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 + 50)

    zone = page.locator("#timeline-zone")
    assert zone.get_attribute("data-splitter-dragging") is not None, (
        "the pane must carry data-splitter-dragging while a drag is in flight"
    )
    assert page.evaluate(
        "getComputedStyle(document.querySelector('#timeline-zone')).transitionDuration"
    ) in ("0s", "0s, 0s"), "the height transition must be suppressed mid-drag"

    page.mouse.up()
    assert zone.get_attribute("data-splitter-dragging") is None, (
        "the attribute must be dropped on release, or discrete collapses stop animating"
    )
    page.close()


# ---------------------------------------------------------------------------
# Collapse restores the size held before collapsing
# ---------------------------------------------------------------------------


def test_double_click_collapses_and_restores_the_prior_height(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The gesture is only useful if it restores where you left it."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    _drag(page, "#timeline-handle", dy=60)
    chosen = _height(page, "#timeline-zone")

    page.locator("#timeline-handle").dblclick()
    page.wait_for_timeout(TRANSITION_SETTLE_MS)
    collapsed = _height(page, "#timeline-zone")
    assert collapsed == pytest.approx(28, abs=2), (
        f"expected the timeline to collapse to its 28px header, got {collapsed}"
    )

    page.locator("#timeline-handle").dblclick()
    page.wait_for_timeout(TRANSITION_SETTLE_MS)
    restored = _height(page, "#timeline-zone")
    assert restored == pytest.approx(chosen, abs=4), (
        f"expected the pre-collapse height ~{chosen}px back, got {restored}"
    )
    page.close()


def test_arrow_keys_nudge_on_the_vertical_axis(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """A divider only reachable by pointer is unusable by keyboard."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    page.locator("#timeline-handle").focus()
    before = _height(page, "#timeline-zone")
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(TRANSITION_SETTLE_MS)
    after = _height(page, "#timeline-zone")

    assert after > before, f"ArrowDown should grow the timeline zone, got {before} -> {after}"
    # aria-valuenow is the splitter's own record of what it applied; it must
    # agree with what the box actually measures.
    assert float(page.locator("#timeline-handle").get_attribute("aria-valuenow")) == pytest.approx(
        after, abs=2
    )
    page.close()


# ---------------------------------------------------------------------------
# The host owns the state: presets still win, and a drag still marks custom
# ---------------------------------------------------------------------------


def test_presets_survive_a_reload_and_keep_the_mode_pill_lit(
    chromium_browser: Browser, dashboard_url: str
) -> None:
    """The reason these splitters take `storageKey: null`.

    If each splitter persisted its own record, a stale value would beat a
    freshly applied preset on the next load and the mode pill would read
    "custom" for a layout the operator never customised.
    """
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    page.locator("#mode-inspect").click()
    page.wait_for_timeout(TRANSITION_SETTLE_MS)
    inspect_timeline = _height(page, "#timeline-zone")

    page.reload(wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.querySelector('#timeline-handle')?.hasAttribute('aria-valuenow')",
        timeout=WIRE_TIMEOUT_MS,
    )
    page.wait_for_timeout(TRANSITION_SETTLE_MS)

    assert _height(page, "#timeline-zone") == pytest.approx(inspect_timeline, abs=4), (
        "the Inspect preset's zone sizes did not survive a reload -- a splitter-owned "
        "record is probably overriding the layout record"
    )
    expect(page.locator("#mode-inspect")).to_have_class(re.compile(r"\bon\b"))
    page.close()


def test_a_drag_marks_the_layout_custom(chromium_browser: Browser, dashboard_url: str) -> None:
    """`onCommit` is the only path back into the layout record; prove it fires."""
    page = chromium_browser.new_page(viewport=VIEWPORT)
    _open_dashboard(page, dashboard_url)

    page.locator("#mode-overview").click()
    expect(page.locator("#mode-overview")).to_have_class("on")
    # The preset click animates the zones into place (200ms height transition)
    # and records mode='overview' when it settles. Dragging before that write
    # lands would let it clobber the drag's own mode='custom' record.
    page.wait_for_timeout(TRANSITION_SETTLE_MS)

    _drag(page, "#timeline-handle", dy=40)

    mode = page.evaluate("JSON.parse(localStorage.getItem('dispatch-panel-layout-v1')).mode")
    assert mode == "custom", f"a drag must mark the layout custom, got mode={mode!r}"
    page.close()
