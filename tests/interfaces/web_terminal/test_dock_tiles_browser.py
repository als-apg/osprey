"""One panel per tile and the engine's drop rules, through the real web terminal.

Every test runs against a live web terminal with four live custom services
(``svc-a`` .. ``svc-d``), each a tiny page with an ``<input id="probe">`` so a
test can prove the service takes a click and typed keys; the agent pane's
chat composer is the terminal-side input.

How a drag is made. Playwright's mouse does not raise HTML5 drag-and-drop over
these iframes, so a drag is dispatched in the page with a real
``DataTransfer``: ``dragstart`` on the real source element (a tile's header
bar, found by its panel or by its tile id, or a sidebar row), then
``dragenter``/``dragover`` on whatever element really is at the target point,
then ``drop`` there and ``dragend`` on the source. Nothing raises the drag
shield by hand: every test asserts that the source's own ``dragstart`` raised
it (``.main-container.dock-dragging``, every service iframe ``pointer-events:
none``) and that it is down again afterwards. A sash resize is a real
Playwright pointer gesture.

How a start state is made. Services are opened from the sidebar as a person
would (each row click replaces the panel of the service tile used last, so
each service gets its iframe); the exact start arrangement (A | B | C at
50/30/20, A over B beside C, ...) is then put in the layout holder with
``restore`` — the same entry a saved dock takes on reload — and every tile's
panel is shown once so its iframe is on screen. After every gesture the test
waits for the holder to be idle and the DOM to show exactly the holder's tiles.

What is read back. The holder's layout as a shape (``row[a,column[b,c],terminal]``:
a tile is the key of the one panel it shows), and each tile's on-screen
rectangle. The drop highlight is read during the drag (its kind class and
rectangle) and compared with the resulting tile's rectangle after the drop
(within 2 px). A tile's header and centre are no drop target: a drag there
paints nothing and drops nothing.

The open rule is pinned at the end: a sidebar row click and the agent's open
replace the panel of the service tile used last (the replaced panel keeps its
row and its iframe), a docked panel is focused where it is, a reload brings
the stored arrangement back whole, and a stored tile holding several panels
comes back with the one it showed. Last, the two narrow-tile defects of the
tab release: removing a panel from the rail leaves the other tiles alone, and
a narrow artifacts tile keeps its bar and a working overflow menu.

Run:

    uv run pytest tests/interfaces/web_terminal/test_dock_tiles_browser.py -rs
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from tests.interfaces._browser import open_hub_page, rail_entry, wait_for_dock_settled
from tests.interfaces.web_terminal.test_panels_browser import _live_server

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover - the chromium fixture skips first
    Page = object  # type: ignore[assignment,misc]

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1600, "height": 1000}
KEYS = ["svc-a", "svc-b", "svc-c", "svc-d"]
TERMINAL = "terminal"
#: Rect tolerance between the highlight and the result, and across a reload.
PX = 2
#: The engine's STRIP_PX: the dock's outer drop strips, this deep on each side.
STRIP_PX = 48
#: The engine's EDGE_RATIO: a tile's edge band, as a share of its size.
EDGE_RATIO = 0.31
_SETTLE_S = 8.0

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def S(text: str) -> str:
    """A shape written with one-letter service names: ``a`` is ``svc-a``."""
    return re.sub(r"(?<![\w-])([a-d])(?![\w-])", r"svc-\1", text)


def _tile(tile: str, *tabs: str, showing: str | None = None, size: float | None = None) -> dict:
    keys = [S(t) if len(t) == 1 else t for t in tabs]
    node = {"tile": tile, "tabs": keys, "showing": S(showing) if showing else keys[0]}
    if size is not None:
        node["size"] = size
    return node


def _split(direction: str, *items: dict, size: float | None = None) -> dict:
    node: dict = {"split": direction, "items": list(items)}
    if size is not None:
        node["size"] = size
    return node


#: A | B | C at 50/30/20, D closed (never opened), terminal closed.
FOUR_TASKS_START = _split(
    "row",
    _tile("t1", "a", size=0.5),
    _tile("t2", "b", size=0.3),
    _tile("t3", "c", size=0.2),
)

#: The drop matrix's start: A over B, then C, then the terminal's tile. C's
#: and the terminal's tiles sit where they did when B was a tab behind A.
MATRIX_START = _split(
    "row",
    _split("column", _tile("t1", "a", size=0.5), _tile("t4", "b", size=0.5), size=0.4),
    _tile("t2", "c", size=0.3),
    _tile("t3", TERMINAL, size=0.3),
)

# ---------------------------------------------------------------------------
# The live services
# ---------------------------------------------------------------------------

_SERVICE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__KEY__</title>
<style>html,body{margin:0;height:100%;background:#fff;font:14px sans-serif}
#probe{display:block;margin:8px;width:160px}</style></head>
<body><b>__KEY__</b><input id="probe" autocomplete="off"></body></html>
"""


@contextmanager
def _service(key: str) -> Iterator[str]:
    """A live service for one panel: a page with a probe input."""
    body = _SERVICE_PAGE.replace("__KEY__", key).encode()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-request stderr logging
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@dataclass
class Hub:
    base_url: str
    app: object
    errors: list[str] = field(default_factory=list)


@contextmanager
def _hub(tmp_path: Path) -> Iterator[Hub]:
    """A live web terminal with the four services as custom panels."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with ExitStack() as stack:
        urls = [stack.enter_context(_service(key)) for key in KEYS]
        custom = [
            {"id": key, "label": key.upper(), "url": url, "healthEndpoint": None, "path": "/"}
            for key, url in zip(KEYS, urls, strict=True)
        ]
        base_url, app = stack.enter_context(
            _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=custom)
        )
        app.state.visible_panels = ["artifacts", *KEYS]
        yield Hub(base_url, app)


def _open(browser, hub: Hub) -> Page:
    page = open_hub_page(browser, hub.base_url, viewport=VIEWPORT)
    # The highlight glides between zones; reduced motion reads it at rest.
    page.emulate_media(reduced_motion="reduce")
    page.on("pageerror", lambda e: hub.errors.append(str(e)))
    wait_for_dock_settled(page)
    return page


# ---------------------------------------------------------------------------
# In-page reads
# ---------------------------------------------------------------------------

#: The holder's layout and, per tile, what its header shows: the tile's bar
#: and the one panel it carries (its ``data-panel-key``), with their rects. A
#: holder tile listing several panels (which the policy forbids) shows in the
#: shape as ``[*a|b]`` so a regression reads plainly.
_STATE_JS = """
async () => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const holder = getLayoutHolder();
  const layout = holder?.layout ?? null;
  const shape = (n) => 'tile' in n
    ? (n.tabs.length === 1 ? n.tabs[0]
       : `[${n.tabs.map((k) => (k === n.showing ? '*' : '') + k).join('|')}]`)
    : `${n.split}[${n.items.map(shape).join(',')}]`;
  const held = [];
  const walk = (n) => { if ('tile' in n) held.push(n); else n.items.forEach(walk); };
  if (layout) walk(layout);
  const r = (el) => {
    const b = el.getBoundingClientRect();
    return { x: b.x, y: b.y, width: b.width, height: b.height };
  };
  const root = document.querySelector('#dock-root');
  const tiles = {};
  for (const bar of root.querySelectorAll('.tile-tab[data-tile]')) {
    if (!bar.isConnected) continue;
    const group = bar.closest('.dv-groupview');
    if (!group) continue;
    const content = group.querySelector('.dv-content-container');
    const own = bar.getAttribute('data-panel-key');
    tiles[bar.dataset.tile] = {
      rect: r(group),
      content: content ? r(content) : null,
      bar: r(bar),
      row: r(bar.closest('.dv-tabs-and-actions-container') ?? bar),
      tabs: [own],
      showing: own,
    };
  }
  const shielded = [...document.querySelectorAll('.dock-iframe-overlay iframe')]
    .some((f) => f.style.pointerEvents === 'none');
  return {
    layout,
    shape: layout ? shape(layout) : null,
    held: held.map((t) => ({ tile: t.tile, tabs: t.tabs, showing: t.showing })),
    tiles,
    dock: r(root),
    dragLive: !!holder?.dragLive,
    dragging: !!document.querySelector('.main-container.dock-dragging'),
    shielded,
    previews: document.querySelectorAll('.dock-drop-preview__rect').length,
  };
}
"""

#: Put a layout in the holder, then show every tile's panel once so each has
#: its iframe on screen, as it would be after a person opened it.
_RESTORE_JS = """
async (layout) => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const { focusPanel } = await import('/static/js/dock-iframe.js');
  const holder = getLayoutHolder();
  if (!holder) return 'no holder';
  if (!holder.restore(layout)) return 'refused';
  const tiles = [];
  const walk = (n) => { if ('tile' in n) tiles.push(n); else n.items.forEach(walk); };
  walk(layout);
  for (const tile of tiles) if (tile.showing !== 'terminal') focusPanel(tile.showing);
  return 'ok';
}
"""

#: One drag, dispatched in the page with a real DataTransfer. The source's own
#: dragstart raises the shield; nothing here touches it. A `tab` source is the
#: bar of the tile showing that panel and a `tile` source the bar of that tile:
#: both are dockview's own header drag, which the bridge reads as the tile. A
#: `rail` source is the sidebar row.
_DRAG_JS = r"""
async ({ source, points, drop, leave }) => {
  const sleep = (t) => new Promise((r) => setTimeout(r, t));
  const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
  const root = document.querySelector('#dock-root');
  let el = null;
  if (source.tab) el = root.querySelector(`.tile-tab[data-panel-key="${CSS.escape(source.tab)}"]`)?.closest('.dv-tab') ?? null;
  else if (source.tile) el = root.querySelector(`.tile-tab[data-tile="${CSS.escape(source.tile)}"]`)?.closest('.dv-tab') ?? null;
  else if (source.rail) el = document.querySelector(`button.panel-rail-button[data-panel-id="${CSS.escape(source.rail)}"]`);
  if (!el) return { error: `no source element for ${JSON.stringify(source)}` };
  const dataTransfer = new DataTransfer();
  const init = (x, y) => ({ bubbles: true, cancelable: true, composed: true, dataTransfer, clientX: x, clientY: y });
  const s = el.getBoundingClientRect();
  const sx = s.x + s.width / 2;
  const sy = s.y + s.height / 2;
  el.dispatchEvent(new DragEvent('dragstart', init(sx, sy)));
  const shieldUp = () => !!document.querySelector('.main-container.dock-dragging');
  let shieldRose = false;
  for (let i = 0; i < 60 && !shieldRose; i += 1) {
    if (shieldUp()) shieldRose = true; else await sleep(16);
  }
  // Whether the preview layer paints over the service iframes: both are
  // children of .main-container, so the higher z-index wins, then the later one.
  const paintsAboveIframes = (layer) => {
    const overlay = document.querySelector('.dock-iframe-overlay');
    if (!overlay || !layer) return true;
    if (overlay.parentElement !== layer.parentElement) return false;
    const z = (el) => Number.parseInt(getComputedStyle(el).zIndex, 10) || 0;
    if (z(layer) !== z(overlay)) return z(layer) > z(overlay);
    return !!(overlay.compareDocumentPosition(layer) & Node.DOCUMENT_POSITION_FOLLOWING);
  };
  const iframes = [...document.querySelectorAll('.dock-iframe-overlay iframe')];
  const iframesShielded = iframes.every((f) => f.style.pointerEvents === 'none');
  let last = null;
  let hit = null;
  let preview = null;
  let point = { x: sx, y: sy };
  for (const p of points) {
    point = p;
    const target = document.elementFromPoint(p.x, p.y);
    if (!target) continue;
    if (target !== last) {
      last?.dispatchEvent(new DragEvent('dragleave', init(p.x, p.y)));
      target.dispatchEvent(new DragEvent('dragenter', init(p.x, p.y)));
      last = target;
    }
    target.dispatchEvent(new DragEvent('dragover', init(p.x, p.y)));
    await frame();
    target.dispatchEvent(new DragEvent('dragover', init(p.x, p.y)));
    await frame();
    hit = {
      inDock: root.contains(target),
      sash: !!target.closest('.dv-sash'),
      terminal: !!target.closest('.dv-groupview')?.querySelector('.tile-tab-terminal'),
      cls: String(target.className || target.tagName),
    };
    const box = document.querySelector('.dock-drop-preview__rect');
    if (box) {
      const b = box.getBoundingClientRect();
      const kind = [...box.classList].find((c) => c.startsWith('dock-drop-preview__rect--'))?.slice(25) ?? null;
      const ph = box.querySelector('.dock-drop-preview__placeholder');
      const cs = ph ? getComputedStyle(ph) : null;
      preview = {
        kind,
        rect: { x: b.x, y: b.y, width: b.width, height: b.height },
        look: cs ? [cs.borderTopStyle, cs.borderTopWidth, cs.top, cs.left].join(' ') : null,
        above: paintsAboveIframes(box.parentElement),
      };
    } else {
      preview = null;
    }
  }
  let previewAfterLeave = null;
  if (leave) {
    // Leave the dock for the sidebar while still dragging.
    const rail = document.querySelector('#sidebar') ?? document.body;
    const r = rail.getBoundingClientRect();
    const out = { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    last?.dispatchEvent(new DragEvent('dragleave', { ...init(out.x, out.y), relatedTarget: rail }));
    const outside = document.elementFromPoint(out.x, out.y) ?? rail;
    outside.dispatchEvent(new DragEvent('dragenter', init(out.x, out.y)));
    outside.dispatchEvent(new DragEvent('dragover', init(out.x, out.y)));
    await frame();
    await frame();
    previewAfterLeave = document.querySelectorAll('.dock-drop-preview__rect').length;
    last = outside;
    point = out;
  }
  if (drop && last) {
    last.dispatchEvent(new DragEvent('drop', init(point.x, point.y)));
  } else {
    last?.dispatchEvent(new DragEvent('dragleave', init(point.x, point.y)));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  }
  (el.isConnected ? el : document).dispatchEvent(new DragEvent('dragend', init(point.x, point.y)));
  return { shieldRose, iframesShielded, iframeCount: iframes.length, hit, preview, previewAfterLeave };
}
"""


def _state(page: Page) -> dict:
    return page.evaluate(_STATE_JS)


def _consistent(state: dict) -> bool:
    """The DOM shows exactly the holder's tiles, tabs and showing tabs; no drag."""
    if state["dragLive"] or state["dragging"] or state["shielded"] or state["previews"]:
        return False
    held = {t["tile"]: t for t in state["held"]}
    if set(held) != set(state["tiles"]):
        return False
    for tid, tile in held.items():
        dom = state["tiles"][tid]
        if dom["tabs"] != tile["tabs"] or dom["showing"] != tile["showing"]:
            return False
        if dom["rect"]["width"] <= 0 or dom["rect"]["height"] <= 0:
            return False
    return True


def _settle(page: Page) -> dict:
    """Wait until the holder is idle and the dock shows it, stable for two reads."""
    deadline = time.monotonic() + _SETTLE_S
    previous = None
    state = _state(page)
    while time.monotonic() < deadline:
        state = _state(page)
        if _consistent(state):
            key = json.dumps([state["shape"], state["tiles"]], sort_keys=True)
            if key == previous:
                return state
            previous = key
        else:
            previous = None
        page.wait_for_timeout(100)
    raise AssertionError(
        f"the dock never settled on the holder's layout: {json.dumps(state)[:2000]}"
    )


def _settle_until(page: Page, ready) -> dict:
    """Settle repeatedly until ``ready(state)`` holds, within the settle budget.

    Returns the last settled state either way, so the caller's assertions name
    what was actually on screen when the budget ran out.
    """
    deadline = time.monotonic() + _SETTLE_S
    state = _settle(page)
    while not ready(state) and time.monotonic() < deadline:
        page.wait_for_timeout(100)
        state = _settle(page)
    return state


def _restore(page: Page, layout: dict) -> dict:
    assert page.evaluate(_RESTORE_JS, layout) == "ok"
    state = _settle(page)
    for tile in state["held"]:
        if tile["showing"] != TERMINAL:
            expect(_iframe(page, tile["showing"])).to_be_visible(timeout=5_000)
    return _settle(page)


#: Whether the stored layout is the holder's: the same tiles, tabs and showing
#: tabs, and the same sizes up to the holder's own tolerance (dock-apply.js's
#: sameShape and sameSizes). With what was stored and what is held, for the
#: assertion message.
_STORED_JS = """
async () => {
  const { getLayoutHolder, sameShape, sameSizes } = await import('/static/js/dock-apply.js');
  const live = getLayoutHolder()?.layout ?? null;
  const showing = (n) => ('tile' in n ? [n.showing] : n.items.flatMap(showing));
  const values = Object.keys(localStorage)
    .filter((k) => k.includes('osprey-dock-layout-'))
    .map((k) => {
      try { return JSON.parse(localStorage.getItem(k))?.layout ?? null; } catch { return null; }
    });
  const matches = (stored) => !!live && !!stored && sameShape(live, stored)
    && sameSizes(live, stored) && showing(live).join() === showing(stored).join();
  return { stored: values.some(matches), live, values };
}
"""


def _wait_until_stored(page: Page) -> None:
    """Wait until the stored layout is the one on screen, sizes included.

    A reload restores what was stored, and the store is written a debounce after
    a change: a reload taken before that write brings back the older sizes.
    """
    deadline = time.monotonic() + _SETTLE_S
    seen = page.evaluate(_STORED_JS)
    while not seen["stored"] and time.monotonic() < deadline:
        page.wait_for_timeout(100)
        seen = page.evaluate(_STORED_JS)
    assert seen["stored"], f"the stored layout never caught up with the one on screen: {seen}"


def _drag(
    page: Page, source: dict, points: list[dict], *, drop: bool = True, leave: bool = False
) -> dict:
    result = page.evaluate(
        _DRAG_JS, {"source": source, "points": points, "drop": drop, "leave": leave}
    )
    assert "error" not in result, result
    # The source's own dragstart raised the shield, over every service iframe.
    assert result["shieldRose"], f"the drag of {source} raised no shield"
    assert result["iframesShielded"], f"the drag of {source} left an iframe unshielded"
    # The highlight shows over a service tile, not hidden behind its iframe.
    assert result["preview"] is None or result["preview"]["above"], result["preview"]
    return result


def _iframe(page: Page, key: str):
    return page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{key}"]')


def _probe(page: Page, key: str):
    return page.frame_locator(f'.dock-iframe-overlay iframe[data-panel-id="{key}"]').locator(
        "#probe"
    )


def _open_services(page: Page, *letters: str) -> None:
    """Open each service from its sidebar row, as a person does, so it has an iframe.

    Each row click replaces the panel of the service tile used last, so only
    the last one stays docked; the others keep their (hidden) iframes for the
    start layout a test restores next.
    """
    for letter in letters:
        key = S(letter)
        rail_entry(page, key, enabled_only=True).click()
        expect(_iframe(page, key)).to_be_visible(timeout=10_000)
    _settle(page)


# --- geometry -----------------------------------------------------------------


def _docked(state: dict) -> set[str]:
    """Every key the holder's layout shows, the terminal's included."""
    return {key for tile in state["held"] for key in tile["tabs"]}


def _tile_of(state: dict, key: str) -> str:
    for tile in state["held"]:
        if key in tile["tabs"]:
            return tile["tile"]
    raise AssertionError(f"{key} is in no tile: {state['shape']}")


def _rect(state: dict, key: str) -> dict:
    return state["tiles"][_tile_of(state, key)]["rect"]


def _content(state: dict, key: str) -> dict:
    return state["tiles"][_tile_of(state, key)]["content"]


def _centre(state: dict, key: str) -> dict:
    c = _content(state, key)
    return {"x": c["x"] + c["width"] / 2, "y": c["y"] + c["height"] / 2}


def _edge_inset(size: float) -> float:
    """How far in from a tile's side its edge band is hit: inside the band and,
    when the band is deep enough, past the dock strip a tile on the dock's edge carries."""
    band = size * EDGE_RATIO
    return (STRIP_PX + band) / 2 if band > STRIP_PX + 8 else band / 2


def _edge(state: dict, key: str, side: str) -> dict:
    """A point in `side`'s edge band of the tile, clear of the dock's outer strips."""
    c = _content(state, key)
    cx, cy = c["x"] + c["width"] / 2, c["y"] + c["height"] / 2
    inset_y = _edge_inset(c["height"])
    inset_x = _edge_inset(c["width"])
    if side == "bottom":
        return {"x": cx, "y": c["y"] + c["height"] - inset_y}
    if side == "top":
        return {"x": cx, "y": c["y"] + inset_y}
    if side == "left":
        return {"x": c["x"] + inset_x, "y": cy}
    return {"x": c["x"] + c["width"] - inset_x, "y": cy}


def _on_bar(state: dict, key: str, at: float = 24) -> dict:
    """A point on the header row of `key`'s tile: `at` px from its left end, or
    from its right end when negative (over the tile's actions)."""
    row = state["tiles"][_tile_of(state, key)]["row"]
    x = row["x"] + at if at >= 0 else row["x"] + row["width"] + at
    return {"x": x, "y": row["y"] + row["height"] / 2}


def _near_split(state: dict, left: str, right: str) -> dict:
    """A point just left of the split line between `left`'s and `right`'s tiles."""
    a, b = _rect(state, left), _rect(state, right)
    top = max(a["y"], b["y"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    return {"x": a["x"] + a["width"] - 6, "y": (top + bottom) / 2 + 20}


def _dock_bottom(state: dict, key: str) -> dict:
    """The dock's last pixels under `key`: the dock's bottom strip, an outer drop."""
    c = _content(state, key)
    d = state["dock"]
    return {"x": c["x"] + c["width"] / 2, "y": d["y"] + d["height"] - 5}


def _below_bar(state: dict, key: str) -> dict:
    """Just below a tile's bar: the dock's top strip when the tile is in the top row."""
    c = _content(state, key)
    return {"x": c["x"] + c["width"] / 2, "y": c["y"] + 5}


def _half(rect: dict, side: str) -> dict:
    """The half of `rect` on `side`: the highlight of an edge drop."""
    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    if side == "left":
        return {"x": x, "y": y, "width": w / 2, "height": h}
    if side == "right":
        return {"x": x + w / 2, "y": y, "width": w / 2, "height": h}
    if side == "top":
        return {"x": x, "y": y, "width": w, "height": h / 2}
    return {"x": x, "y": y + h / 2, "width": w, "height": h / 2}


def _close(a: dict, b: dict, tol: float = PX) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("x", "y", "width", "height"))


def _pct(state: dict, *keys: str) -> list[float]:
    widths = [_rect(state, k)["width"] for k in keys]
    total = sum(widths)
    return [100 * w / total for w in widths]


def _assert_same_rects(before: dict, after: dict, keys: list[str]) -> None:
    for key in keys:
        assert _close(_rect(before, key), _rect(after, key)), (
            f"{key} changed size: {_rect(before, key)} -> {_rect(after, key)}"
        )


def _drop(page: Page, source: dict, point: dict, *, kind: str | None = None) -> tuple[dict, dict]:
    """Drag `source` to `point` and drop; returns (drag result, settled state)."""
    result = _drag(page, source, [point])
    assert result["preview"] is not None, f"no highlight for {source} at {point}: {result['hit']}"
    if kind is not None:
        assert result["preview"]["kind"] == kind, result
    return result, _settle(page)


def _refused(page: Page, source: dict, point: dict, start: dict) -> None:
    """Drag `source` to `point` and drop: no highlight, nothing changes."""
    result = _drag(page, source, [point])
    assert result["preview"] is None, f"{source} at {point} painted a highlight: {result}"
    after = _settle(page)
    assert after["shape"] == start["shape"], (start["shape"], after["shape"])


def _bar(page: Page, key: str):
    """The header bar of the tile showing `key`."""
    return page.locator(f'#dock-root .tile-tab[data-panel-key="{key}"]')


def _focus_tile(page: Page, key: str) -> dict:
    """Click the title of `key`'s tile — the focus gesture — and settle."""
    _bar(page, key).locator(".tile-tab-title").click()
    return _settle(page)


# --- input --------------------------------------------------------------------


def _type_in_service(page: Page, key: str, text: str) -> None:
    probe = _probe(page, key)
    probe.click(timeout=5_000)
    probe.fill("")
    page.keyboard.type(text)
    expect(probe).to_have_value(text, timeout=5_000)


def _composer(page: Page):
    return page.locator("#operator-container .op-input-area textarea")


def _type_in_terminal(page: Page, text: str) -> None:
    """Typed keys reach the agent pane's composer."""
    target = _composer(page)
    expect(target).to_be_visible(timeout=10_000)
    target.click(timeout=5_000)
    target.fill("")
    page.keyboard.type(text)
    expect(target).to_have_value(text, timeout=5_000)


def _assert_terminal_shown(page: Page, state: dict) -> None:
    """The terminal is its tile's showing tab and nothing covers it."""
    tid = _tile_of(state, TERMINAL)
    assert state["tiles"][tid]["showing"] == TERMINAL, state["shape"]
    c = state["tiles"][tid]["content"]
    covered = page.evaluate(
        "({x, y}) => { const el = document.elementFromPoint(x, y);"
        " return !el || !el.closest('#terminal-container, #operator-container'); }",
        {"x": c["x"] + c["width"] / 2, "y": c["y"] + c["height"] / 2},
    )
    assert not covered, f"the terminal is covered: {state['shape']}"


# ---------------------------------------------------------------------------
# T1-T4
# ---------------------------------------------------------------------------


def test_four_tasks_each_in_the_stated_number_of_drops(tmp_path, chromium_browser):
    """T1-T4 from A | B | C at 50/30/20 with D closed."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")

        # T1: D from the sidebar onto B's bottom edge lands in B's bottom half,
        # exactly where the highlight was; A and C do not move.
        start = _restore(page, FOUR_TASKS_START)
        assert start["shape"] == S("row[a,b,c]")
        assert [round(p) for p in _pct(start, a, b, c)] == [50, 30, 20]
        drag, after = _drop(page, {"rail": d}, _edge(start, b, "bottom"), kind="split")
        assert after["shape"] == S("row[a,column[b,d],c]")
        _assert_same_rects(start, after, [a, c])
        rd = _rect(after, d)
        assert _close(rd, _half(_rect(start, b), "bottom"), 2 * PX), (rd, _rect(start, b))
        assert _close(drag["preview"]["rect"], rd), (drag["preview"], rd)

        # T2: D onto B's centre: a tile holds one panel, so its centre is no
        # target — no highlight, nothing changes.
        start = _restore(page, FOUR_TASKS_START)
        _refused(page, {"rail": d}, _centre(start, b), start)

        # T3: A next to the B|C split line is B's right edge: B | A | C, with A
        # beside B at B's width (A's own tile empties first, so B is wider by then).
        start = _restore(page, FOUR_TASKS_START)
        drag, after = _drop(page, {"tile": "t1"}, _near_split(start, b, c), kind="split")
        assert after["shape"] == S("row[b,a,c]")
        assert abs(_rect(after, b)["width"] - _rect(after, a)["width"]) <= 2 * PX, _pct(
            after, b, a, c
        )

        # T4: the 2x2 grid (A over D, B over C) in two drops.
        start = _restore(page, FOUR_TASKS_START)
        _, mid = _drop(page, {"rail": d}, _edge(start, a, "bottom"), kind="split")
        assert mid["shape"] == S("row[column[a,d],b,c]")
        _, after = _drop(page, {"tile": _tile_of(mid, c)}, _edge(mid, b, "bottom"), kind="split")
        assert after["shape"] == S("row[column[a,d],column[b,c]]")
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Every drop type x every source
# ---------------------------------------------------------------------------

#: (source, target, expected shape) from MATRIX_START: row[column[a,b], c, terminal].
#: Targets are on C's tile — whose header and centre take nothing from any
#: source (None: no target), since a tile holds one panel — or beside it: C's
#: bottom edge, the split line just left of the C|terminal line and the dock
#: side, the dock's bottom under C; the first two are edges of C. A `tab`
#: source is B's bar, a `tile` source A's tile (t1) by its bar, a `rail` source
#: D's sidebar row.
_MATRIX = [
    ("tab", "centre", None),
    ("tab", "tabbar", None),
    ("tab", "edge", "row[a,column[c,b],terminal]"),
    ("tab", "split line", "row[a,c,b,terminal]"),
    ("tab", "dock side", "column[row[a,c,terminal],b]"),
    ("tile", "centre", None),
    ("tile", "tabbar", None),
    ("tile", "edge", "row[b,column[c,a],terminal]"),
    ("tile", "split line", "row[b,c,a,terminal]"),
    ("tile", "dock side", "column[row[b,c,terminal],a]"),
    ("rail", "centre", None),
    ("rail", "tabbar", None),
    ("rail", "edge", "row[column[a,b],column[c,d],terminal]"),
    ("rail", "split line", "row[column[a,b],c,d,terminal]"),
    ("rail", "dock side", "column[row[column[a,b],c,terminal],d]"),
]
#: The placeholder every highlight paints: a 2 px solid stroke, inset 4 px.
_PLACEHOLDER = "solid 2px 4px 4px"


def test_every_drop_type_for_every_source(tmp_path, chromium_browser):
    """Centre, tab bar, edge, split line and dock side, for a tab, a tile and a sidebar row.

    A tile's header and centre are no target for any source: nothing is
    painted and nothing changes. Every other drop splits: the highlight
    outlines the result within 2 px (a moved tile, whose own space goes to a
    neighbour first, can land wider), every highlight is the same placeholder,
    none stays painted after the drop, the shield goes down, and a moved tile
    keeps its id. A self-drop shows nothing and changes nothing; a drag that
    leaves the dock leaves no highlight behind.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        looks: set[str] = set()
        for source_kind, target, want in _MATRIX:
            start = _restore(page, MATRIX_START)
            assert start["shape"] == S("row[column[a,b],c,terminal]")
            source = {"tab": {"tab": b}, "tile": {"tile": "t1"}, "rail": {"rail": d}}[source_kind]
            moved = {"tab": b, "tile": a, "rail": d}[source_kind]
            if target == "centre":
                point = _centre(start, c)
            elif target == "tabbar":
                point = _on_bar(start, c)
            elif target == "edge":
                point = _edge(start, c, "bottom")
            elif target == "split line":
                point = _near_split(start, c, TERMINAL)
            else:
                point = _dock_bottom(start, c)
            case = f"{source_kind} -> {target}"
            if want is None:
                _refused(page, source, point, start)
                continue
            drag, after = _drop(page, source, point, kind="split")
            assert after["shape"] == S(want), f"{case}: {after['shape']} != {S(want)}"
            # A moved tile's own space goes to a neighbour first, so the result
            # can be wider than the highlight. B's leaves only A's column, and
            # a row's takes none, so theirs match it.
            if source_kind != "tile":
                assert _close(drag["preview"]["rect"], _rect(after, moved)), (
                    f"{case}: highlight {drag['preview']['rect']} != result {_rect(after, moved)}"
                )
            if source_kind == "tile":
                assert _tile_of(after, a) == "t1", f"{case}: the moved tile changed its id"
            looks.add(drag["preview"]["look"])
            assert after["previews"] == 0 and not after["shielded"]

        # Every highlight is the one placeholder.
        assert looks == {_PLACEHOLDER}, looks

        # A self-drop (a tile on its own edge) shows nothing and changes nothing.
        start = _restore(page, MATRIX_START)
        drag = _drag(page, {"tile": "t2"}, [_edge(start, c, "bottom")])
        assert drag["preview"] is None, drag
        after = _settle(page)
        assert after["shape"] == start["shape"]
        _assert_same_rects(start, after, [a, b, c, TERMINAL])

        # A drag that leaves the dock leaves no highlight, and a cancelled drag changes nothing.
        drag = _drag(page, {"tab": b}, [_edge(start, c, "bottom")], drop=False, leave=True)
        assert drag["preview"] is not None, "the edge painted no highlight before the leave"
        assert drag["previewAfterLeave"] == 0, "a highlight stayed painted after leaving the dock"
        after = _settle(page)
        assert after["shape"] == start["shape"] and after["previews"] == 0
        assert hub.errors == []


def test_the_agent_pane_is_always_its_own_tile(tmp_path, chromium_browser):
    """Nothing drops into the agent pane's tile and it drops into no other; its edges split."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(page, MATRIX_START)

        # Onto the agent pane: a panel's bar, a tile and a row have no target
        # on its header or centre.
        for source in ({"tab": b}, {"tile": "t2"}, {"rail": d}):
            _refused(page, source, _centre(start, TERMINAL), start)
            _refused(page, source, _on_bar(start, TERMINAL), start)
        # Its edge splits it, the dropped item on that side.
        drag, after = _drop(page, {"tab": b}, _edge(start, TERMINAL, "bottom"), kind="split")
        assert after["shape"] == S("row[a,c,column[terminal,b]]")
        assert _close(drag["preview"]["rect"], _rect(after, b)), (drag["preview"], _rect(after, b))
        _assert_terminal_shown(page, after)

        # From the agent pane: its tile has no target on another tile's header
        # or centre, and splits an edge.
        start = _restore(page, MATRIX_START)
        agent_tile = _tile_of(start, TERMINAL)
        _refused(page, {"tile": agent_tile}, _centre(start, c), start)
        _refused(page, {"tile": agent_tile}, _on_bar(start, c), start)
        _, after = _drop(page, {"tile": agent_tile}, _edge(start, c, "bottom"), kind="split")
        assert after["shape"] == S("row[column[a,b],column[c,terminal]]")
        _assert_terminal_shown(page, after)
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Headers, the strip below a top bar, sashes, the clamp
# ---------------------------------------------------------------------------


def test_just_below_a_top_bar_is_the_dock_top(tmp_path, chromium_browser):
    """A tile's top edge splits it; the pixels just under a top-row bar are the dock's top strip."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, c, d = (S(x) for x in "acd")
        three = _split(
            "row",
            _tile("t1", "a", size=0.6),
            _tile("t2", "c", size=0.2),
            _tile("t3", TERMINAL, size=0.2),
        )

        # A tile's top edge, below the dock's top strip: its top half.
        start = _restore(page, three)
        drag, after = _drop(page, {"rail": d}, _edge(start, a, "top"), kind="split")
        assert after["shape"] == S("row[column[d,a],c,terminal]")
        rd = _rect(after, d)
        assert _close(rd, _half(_rect(start, a), "top"), 2 * PX), (rd, _rect(start, a))
        assert _close(drag["preview"]["rect"], rd)

        # Just below a top-row bar is the dock's top strip: a row across the dock.
        start = _restore(page, three)
        drag, after = _drop(page, {"rail": d}, _below_bar(start, a), kind="split")
        assert after["shape"] == S("column[d,row[a,c,terminal]]")
        assert _close(drag["preview"]["rect"], _rect(after, d))

        # The same below the terminal's bar, for a panel's bar.
        start = _restore(page, three)
        _, after = _drop(page, {"tab": c}, _below_bar(start, TERMINAL), kind="split")
        assert after["shape"] == S("column[c,row[a,terminal]]")
        assert hub.errors == []


def test_no_point_on_a_tile_header_is_a_target(tmp_path, chromium_browser):
    """The title, the free space and the actions of a service tile's header take nothing.

    A tile holds one panel, so its whole header row is no target for a
    panel's bar, a tile or a sidebar row: nothing is painted, nothing changes.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        b, c, d = (S(x) for x in "bcd")
        start = _restore(page, MATRIX_START)
        row = start["tiles"][_tile_of(start, c)]["row"]
        assert row["width"] > 200, row
        points = [_on_bar(start, c), _on_bar(start, c, row["width"] / 2), _on_bar(start, c, -8)]
        for source in ({"tab": b}, {"tile": "t1"}, {"rail": d}):
            for point in points:
                _refused(page, source, point, start)
        assert hub.errors == []


def test_next_to_a_split_line_is_the_hovered_tiles_half(tmp_path, chromium_browser):
    """Next to the line between two columns, the drop lands in the hovered tile's half."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(
            page,
            _split(
                "row",
                _split("column", _tile("t1", "a"), _tile("t2", "b"), size=0.5),
                _split("column", _tile("t3", "c"), _tile("t4", TERMINAL), size=0.5),
            ),
        )
        assert start["shape"] == S("row[column[a,b],column[c,terminal]]")
        drag, after = _drop(page, {"rail": d}, _near_split(start, a, c), kind="split")
        assert after["shape"] == S("row[column[row[a,d],b],column[c,terminal]]")
        rd = _rect(after, d)
        assert _close(rd, _half(_rect(start, a), "right"), 2 * PX), (rd, _rect(start, a))
        assert _close(drag["preview"]["rect"], rd)
        _assert_same_rects(start, after, [b, c, TERMINAL])
        assert hub.errors == []


def test_sash_resize_then_t3_keeps_the_resized_ratio(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, FOUR_TASKS_START)

        # A real pointer gesture on the A|B sash: A shrinks by ~200 px.
        ra = _rect(start, a)
        x = ra["x"] + ra["width"]
        y = ra["y"] + ra["height"] / 2
        sash_ok = page.evaluate(
            "({x, y}) => !!document.elementFromPoint(x, y)?.closest('.dv-sash')", {"x": x, "y": y}
        )
        assert sash_ok, "no sash at the A|B border"
        page.mouse.move(x, y)
        page.mouse.down()
        for step in range(1, 11):
            page.mouse.move(x - 20 * step, y)
        page.mouse.up()
        resized = _settle(page)
        assert _rect(resized, a)["width"] < ra["width"] - 150, "the sash did not move"
        sizes = [item["size"] for item in resized["layout"]["items"]]
        widths = [_rect(resized, k)["width"] for k in (a, b, c)]
        for size, width in zip(sizes, widths, strict=True):
            assert abs(size - width / sum(widths)) <= 0.01, (sizes, widths)

        # T3 after the resize: A's space goes to B, A takes B's right half, and C
        # keeps the width the sash gave it.
        _, after = _drop(page, {"tile": "t1"}, _near_split(resized, b, c), kind="split")
        assert after["shape"] == S("row[b,a,c]")
        got = _pct(after, b, a, c)
        assert abs(got[0] - got[1]) <= 1, got
        assert abs(_rect(after, c)["width"] - _rect(resized, c)["width"]) <= PX, got
        assert hub.errors == []


def test_a_tile_too_narrow_to_halve_is_no_target(tmp_path, chromium_browser):
    """B is 60 px wide: halves of 30 px are under 40, so its sides are its centre.

    A tile's centre is no target, so the drop paints nothing and nothing moves.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        probe = _restore(page, FOUR_TASKS_START)
        narrow = 60 / probe["dock"]["width"]
        start = _restore(
            page,
            _split(
                "row",
                _tile("t1", "a", size=0.5 - narrow / 2),
                _tile("t2", "b", size=narrow),
                _tile("t3", "c", size=0.5 - narrow / 2),
            ),
        )
        rb = _rect(start, b)
        cb = _content(start, b)
        assert 50 <= rb["width"] < 80, rb
        # B's left side, at mid-height: clear of its top and bottom edges.
        point = {"x": cb["x"] + 8, "y": cb["y"] + cb["height"] / 2}
        _refused(page, {"tile": "t3"}, point, start)
        _assert_same_rects(start, _settle(page), [a, b, c])
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Input after drags and sidebar switches
# ---------------------------------------------------------------------------


def test_services_and_terminal_take_input_after_a_drag_and_a_panel_switch(
    tmp_path, chromium_browser
):
    """Keys reach the services and the terminal after a drop and after a sidebar switch.

    The switch is the sidebar's: D's row puts D in the tile C showed, and C's
    row puts C back there with what was typed into it before — its iframe was
    kept, not reloaded.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, c, d = (S(x) for x in "acd")
        start = _restore(page, MATRIX_START)

        _drop(page, {"tile": _tile_of(start, c)}, _edge(start, a, "bottom"), kind="split")
        _type_in_service(page, a, "after-drag-a")
        _type_in_service(page, c, "after-drag-c")
        _type_in_terminal(page, "afterdrag1")

        # C's tile is used last: D's row replaces C there.
        c_tile = _tile_of(_focus_tile(page, c), c)
        rail_entry(page, d, enabled_only=True).click()
        state = _settle_until(page, lambda s: d in _docked(s))
        assert _tile_of(state, d) == c_tile and c not in _docked(state), state["shape"]
        expect(_iframe(page, c)).to_be_hidden(timeout=5_000)
        _type_in_service(page, d, "shown-d")

        # C's row brings C back into the same tile, its text intact.
        rail_entry(page, c, enabled_only=True).click()
        state = _settle_until(page, lambda s: c in _docked(s))
        assert _tile_of(state, c) == c_tile and d not in _docked(state), state["shape"]
        expect(_iframe(page, d)).to_be_hidden(timeout=5_000)
        expect(_probe(page, c)).to_have_value("after-drag-c", timeout=5_000)
        _type_in_service(page, c, "shown-again-c")
        _type_in_terminal(page, "afterswitch2")
        assert hub.errors == []


def test_a_replaced_panel_stays_off_screen_after_a_drop_elsewhere(tmp_path, chromium_browser):
    """D replaces A in A's tile; a later drop re-lays the dock and A stays hidden under D."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, c, d = (S(x) for x in "acd")
        start = _restore(page, MATRIX_START)
        _focus_tile(page, a)
        rail_entry(page, d, enabled_only=True).click()
        shown = _settle_until(page, lambda s: d in _docked(s))
        assert shown["shape"] == S("row[column[d,b],c,terminal]")
        _drop(page, {"tile": _tile_of(start, c)}, _edge(start, TERMINAL, "bottom"), kind="split")
        after = _settle(page)
        assert after["shape"] == S("row[column[d,b],column[terminal,c]]")
        expect(_iframe(page, d)).to_be_visible()
        expect(_iframe(page, a)).to_be_hidden()
        _type_in_service(page, d, "still-d")
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Close and the way back, the split-from neighbour
# ---------------------------------------------------------------------------


def test_close_and_the_way_back_through_the_sidebar_row(tmp_path, chromium_browser):
    """A tile's × closes it, one neighbour takes its space, and the panel's row brings it back."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, MATRIX_START)

        def close_tile(key: str) -> dict:
            bar = _bar(page, key)
            bar.hover()
            bar.locator(".tile-tab-close").click()
            return _settle_until(page, lambda s: key not in _docked(s))

        # B's tile, under A's: A takes the column back, C and the terminal stay.
        state = close_tile(b)
        assert state["shape"] == S("row[a,c,terminal]")
        _assert_same_rects(start, state, [c, TERMINAL])
        assert _rect(state, a)["height"] > _rect(start, a)["height"] + 100, state["tiles"]
        rail_entry(page, b, enabled_only=True).click()
        state = _settle_until(page, lambda s: b in _docked(s))
        assert b in _docked(state), state["shape"]
        _type_in_service(page, b, "back-b")

        # C's tile: one neighbour grows, then back from the sidebar.
        before = _restore(page, MATRIX_START)
        state = close_tile(c)
        assert state["shape"] == S("row[column[a,b],terminal]")
        grew = [k for k in (a, TERMINAL) if not _close(_rect(before, k), _rect(state, k))]
        assert len(grew) == 1, (before["tiles"], state["tiles"])
        rail_entry(page, c, enabled_only=True).click()
        state = _settle_until(page, lambda s: c in _docked(s))
        assert c in _docked(state), state["shape"]
        _type_in_service(page, c, "back-c")

        # The agent pane: closed, then back from its sidebar row, and it takes keys.
        _restore(page, MATRIX_START)
        state = close_tile(TERMINAL)
        assert TERMINAL not in _docked(state), state["shape"]
        rail_entry(page, TERMINAL).click()
        state = _settle_until(page, lambda s: TERMINAL in _docked(s))
        assert TERMINAL in _docked(state), state["shape"]
        _assert_terminal_shown(page, state)
        _type_in_terminal(page, "terminalback")
        assert hub.errors == []


def test_closing_a_split_tile_gives_its_space_back_to_the_tile_it_split_from(
    tmp_path, chromium_browser
):
    """The leaving tile's space goes to the one neighbour it was split from."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(
            page,
            _split(
                "row",
                _tile("t1", "a", size=1 / 3),
                _tile("t2", "b", size=1 / 3),
                _tile("t3", "c", size=1 / 3),
            ),
        )
        # D splits C from its left edge: it sits between B and C, from C.
        _, split = _drop(page, {"rail": d}, _edge(start, c, "left"), kind="split")
        assert split["shape"] == S("row[a,b,d,c]")
        _assert_same_rects(start, split, [a, b])
        bar = page.locator(f'#dock-root .tile-tab[data-tile="{_tile_of(split, d)}"]:visible')
        bar.hover()
        bar.locator(".tile-tab-close").click()
        after = _settle(page)
        assert after["shape"] == S("row[a,b,c]")
        # C, the tile D split from, takes the space back, not B (the previous one).
        _assert_same_rects(start, after, [a, b, c])
        assert hub.errors == []


# ---------------------------------------------------------------------------
# The open rule
# ---------------------------------------------------------------------------


def test_an_open_replaces_the_panel_in_the_service_tile_used_last(tmp_path, chromium_browser):
    """A row click and the agent's open replace what the service tile used last shows.

    Nothing resizes and the Agent pane is never the target. The replaced
    panel keeps its row, which brings it back the same way with the state its
    iframe held; a row click on a docked panel focuses it where it is, so the
    next open replaces that one.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(page, MATRIX_START)
        _type_in_service(page, c, "kept-c")

        # C's tile is the one used last, then the operator types in the Agent
        # pane: D's row replaces C in C's tile, never the Agent pane.
        _focus_tile(page, c)
        _type_in_terminal(page, "beforeopen")
        rail_entry(page, d, enabled_only=True).click()
        after = _settle_until(page, lambda s: d in _docked(s))
        assert after["shape"] == S("row[column[a,b],d,terminal]")
        assert _tile_of(after, d) == "t2", after["held"]
        _assert_same_rects(start, after, [a, b, TERMINAL])
        assert _close(_rect(after, d), _rect(start, c)), (_rect(start, c), _rect(after, d))
        expect(_iframe(page, d)).to_be_visible(timeout=5_000)
        expect(_iframe(page, c)).to_be_hidden(timeout=5_000)
        expect(rail_entry(page, c)).to_have_count(1)
        _type_in_service(page, d, "open-d")

        # C's row brings C back the same way, with what was typed into it.
        rail_entry(page, c, enabled_only=True).click()
        after = _settle_until(page, lambda s: c in _docked(s))
        assert after["shape"] == S("row[column[a,b],c,terminal]")
        assert _tile_of(after, c) == "t2", after["held"]
        expect(_probe(page, c)).to_have_value("kept-c", timeout=5_000)

        # A docked panel's row focuses it where it is: nothing moves, and the
        # next open replaces it rather than C.
        rail_entry(page, a, enabled_only=True).click()
        after = _settle(page)
        assert after["shape"] == S("row[column[a,b],c,terminal]")
        _assert_same_rects(start, after, [a, b, c, TERMINAL])
        rail_entry(page, d, enabled_only=True).click()
        after = _settle_until(page, lambda s: d in _docked(s))
        assert after["shape"] == S("row[column[d,b],c,terminal]")
        assert _tile_of(after, d) == "t1", after["held"]

        # The agent's open lands by the same rule as the row click.
        start = _restore(page, MATRIX_START)
        _focus_tile(page, c)
        r = requests.post(
            f"{hub.base_url}/api/panel-op/open",
            json={"request_id": "tiles-open-1", "member": d},
            timeout=10,
        )
        assert r.status_code == 200, r.text
        expect(_bar(page, d)).to_have_count(1, timeout=10_000)
        after = _settle(page)
        assert after["shape"] == S("row[column[a,b],d,terminal]")
        _assert_same_rects(start, after, [a, b, TERMINAL])
        _type_in_service(page, d, "agent-d")
        assert hub.errors == []


def test_a_reload_brings_the_stored_layout_back_whole(tmp_path, chromium_browser):
    """Tiles, the panel each shows and the sizes come back as they were saved."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        _restore(page, MATRIX_START)
        # The reload brings back what was stored, so the stored layout must be
        # the one on screen first, sizes included: the persist is debounced, and
        # a reload taken before its last write restores older sizes. The state
        # the reload must bring back is read once the store has caught up.
        _wait_until_stored(page)
        before = _settle(page)

        page.reload()
        wait_for_dock_settled(page)
        for key in (a, b, c):
            expect(_iframe(page, key)).to_be_visible(timeout=10_000)
        # The stored sizes are applied after the boot layout; under load the dock
        # can be still at an intermediate size for a moment, so wait on the
        # rects themselves rather than on one settled read.
        after = _settle_until(
            page,
            lambda state: (
                state["shape"] == before["shape"]
                and all(
                    _close(_rect(before, key), _rect(state, key), 2 * PX)
                    for key in (a, b, c, TERMINAL)
                )
            ),
        )
        assert after["shape"] == S("row[column[a,b],c,terminal]") == before["shape"]
        for key in (a, b, c, TERMINAL):
            assert _close(_rect(before, key), _rect(after, key), 2 * PX), (
                key,
                _rect(before, key),
                _rect(after, key),
            )
        _type_in_service(page, a, "after-reload")
        assert hub.errors == []


#: Replace the persisted layout's tree with `layout`, keeping the stored value's
#: version and panel records: what a dock saved by an earlier release reads as.
_STORE_LAYOUT_JS = """(layout) => {
  const key = Object.keys(localStorage).find((k) => k.includes('osprey-dock-layout-'));
  if (!key) return null;
  const value = JSON.parse(localStorage.getItem(key));
  value.layout = layout;
  localStorage.setItem(key, JSON.stringify(value));
  return key;
}"""


def test_a_stored_tile_holding_several_panels_comes_back_with_the_one_it_showed(
    tmp_path, chromium_browser
):
    """A dock saved with A and B in one tile, B in front, reloads with B alone in that tile.

    A leaves the dock and keeps its sidebar row; C and the terminal come back
    as they were.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        _restore(page, MATRIX_START)
        _wait_until_stored(page)
        # Let any trailing persist land before the stored value is rewritten.
        page.wait_for_timeout(400)
        crowded = _split(
            "row",
            _tile("t1", "a", "b", showing="b", size=0.4),
            _tile("t2", "c", size=0.3),
            _tile("t3", TERMINAL, size=0.3),
        )
        assert page.evaluate(_STORE_LAYOUT_JS, crowded), "no stored layout to rewrite"

        page.reload()
        wait_for_dock_settled(page)
        for key in (b, c):
            expect(_iframe(page, key)).to_be_visible(timeout=10_000)
        after = _settle_until(page, lambda s: s["shape"] == S("row[b,c,terminal]"))
        assert after["shape"] == S("row[b,c,terminal]"), after["shape"]
        assert _tile_of(after, b) == "t1", after["held"]
        expect(_iframe(page, a)).to_be_hidden()
        expect(rail_entry(page, a)).to_have_count(1)
        _type_in_service(page, b, "migrated-b")
        assert hub.errors == []


# ---------------------------------------------------------------------------
# The narrow-tile defects of the tab release
# ---------------------------------------------------------------------------


def test_removing_a_panel_from_the_rail_leaves_the_other_tiles_alone(tmp_path, chromium_browser):
    """ "Remove from rail" on a tile's own menu closes that tile and touches no other.

    The tab release's defect: removing the panel the operator was on handed
    the view to a fallback panel opened INTO a tile that still showed one, and
    that tile's header collapsed. Now C's tile closes (its space goes to the
    tile it split from, the terminal's), the view goes to a panel already on
    screen, and A's and B's tiles keep their panel, their visible iframe and
    their rectangle; no other panel (artifacts) is opened.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(
            page,
            _split(
                "row",
                _tile("t1", "a", size=0.25),
                _tile("t2", "b", size=0.25),
                {**_tile("t3", "c", size=0.2), "from": "t4"},
                _tile("t4", TERMINAL, size=0.3),
            ),
        )
        assert start["shape"] == S("row[a,b,c,terminal]")
        # C is the panel the operator is on.
        _focus_tile(page, c)
        expect(page.locator(f'button.panel-rail-button[data-panel-id="{c}"].active')).to_have_count(
            1, timeout=5_000
        )

        _bar(page, c).locator(".tile-tab-title").click(button="right")
        menu = page.locator(".rail-context-menu")
        expect(menu).to_have_count(1, timeout=5_000)
        menu.locator(".rail-context-item", has_text="Remove from rail").click()

        expect(rail_entry(page, c)).to_have_count(0, timeout=10_000)
        after = _settle_until(page, lambda s: c not in _docked(s))
        assert after["shape"] == S("row[a,b,terminal]"), after["shape"]
        _assert_same_rects(start, after, [a, b])
        for key in (a, b):
            expect(_iframe(page, key)).to_be_visible(timeout=5_000)
            assert _iframe(page, key).evaluate("(f) => f.style.display") != "none", key
        expect(_iframe(page, c)).to_be_hidden()
        # The view went to a panel already on screen, in place.
        page.wait_for_timeout(500)
        later = _settle(page)
        assert later["shape"] == after["shape"], later["shape"]
        _assert_same_rects(start, later, [a, b])
        _type_in_service(page, a, "after-remove-a")
        assert hub.errors == []


#: Contribute a hub-drawn header that has to fold: a crumb, a pager and two
#: labelled buttons. Every action it dispatches is recorded.
_ARTIFACTS_HEADER_JS = """async (title) => {
  const { contributeLocal } = await import('/static/js/tile-header-contrib.js');
  const { headerItems } = await import('/static/js/artifacts-pane-header.js');
  const shown = { id: 'x', filename: 'plot.png', title };
  window.__picked = [];
  contributeLocal('artifacts', headerItems(shown), (id) => { window.__picked.push(id); });
}"""

#: The tile ⋯ row for each panel icon the artifacts header can fold (an icon's
#: row is named by its title).
_ARTIFACT_FOLD_LABELS = {
    "icon:info": "Artifact details",
    "icon:download": "Download",
}


def test_a_narrow_artifacts_tile_keeps_its_bar_and_a_working_tile_menu(tmp_path, chromium_browser):
    """In a narrow tile the artifacts bar spans its tile and the tile ⋯ holds what folded.

    The tab release's defect: the artifacts header collapsed the header of a
    tile holding several panels to nothing and folded every control away, and
    the menu that held them closed itself as the clipped region scrolled to
    the focused button. With one panel per tile the bar is the tile's whole
    header. The panel icons the width cannot hold (Info, Download) take class
    `contrib-hidden` and are rows under the tile menu's "This panel" heading,
    a real click on ⋯ keeps the menu open, and a row runs the control it
    stands for.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        # Artifacts has no rail entry; the boot layout docks it.
        expect(_iframe(page, "artifacts")).to_be_visible(timeout=10_000)
        _open_services(page, "a")
        _restore(
            page,
            _split(
                "row",
                _tile("t1", "artifacts", size=0.12),
                _tile("t2", "a", size=0.58),
                _tile("t3", TERMINAL, size=0.3),
            ),
        )
        # The sidebar's own artifacts header has landed before this one replaces it.
        page.wait_for_timeout(500)
        title = "A fairly long artifact name"
        page.evaluate(_ARTIFACTS_HEADER_JS, title)
        bar = _bar(page, "artifacts")
        expect(bar.locator("[data-contrib-key='icon:info']")).to_be_attached(timeout=5_000)

        state = _settle(page)
        tile = state["tiles"][_tile_of(state, "artifacts")]
        assert abs(tile["bar"]["width"] - tile["row"]["width"]) <= PX, tile

        folded = bar.locator("[data-contrib-key].contrib-hidden")
        expect(folded.first).to_be_attached(timeout=5_000)
        keys = folded.evaluate_all("els => els.map(e => e.dataset.contribKey)")
        assert keys and all(key in _ARTIFACT_FOLD_LABELS for key in keys), keys

        bar.locator(".tile-tab-more").click()
        page.wait_for_timeout(300)
        menu = page.locator(".rail-context-menu")
        expect(menu).to_have_count(1)
        expect(menu.locator(".rail-context-heading", has_text="This panel")).to_have_count(1)
        rows = menu.locator(".rail-context-item")
        labels = [text.strip() for text in menu.locator(".rail-context-label").all_inner_texts()]
        for key in keys:
            assert _ARTIFACT_FOLD_LABELS[key] in labels, (labels, keys)

        key = keys[0]
        row = rows.filter(
            has=page.locator(".rail-context-label", has_text=_ARTIFACT_FOLD_LABELS[key])
        )
        expect(row).to_have_count(1)
        expect(row).not_to_have_attribute("aria-disabled", "true")
        row.click()
        expect(menu).to_have_count(0, timeout=5_000)
        assert page.evaluate("() => window.__picked") == [key.split(":", 1)[1]]
        assert hub.errors == []
