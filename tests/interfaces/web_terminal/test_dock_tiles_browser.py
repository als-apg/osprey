"""Tabs in the tile header and the engine's drop rules, through the real web terminal.

Every test runs against a live web terminal with four live custom services
(``svc-a`` .. ``svc-d``), each a tiny page with an ``<input id="probe">`` so a
test can prove the service takes a click and typed keys; the agent pane's
chat composer is the terminal-side input.

How a drag is made. Playwright's mouse does not raise HTML5 drag-and-drop over
these iframes, so a drag is dispatched in the page with a real
``DataTransfer``: ``dragstart`` on the real source element (a header tab
item, a tile's header bar, or a sidebar row), then ``dragenter``/``dragover``
on whatever element really is at the target point, then ``drop`` there and
``dragend`` on the source. Nothing raises the drag shield by hand: every test
asserts that the source's own ``dragstart`` raised it
(``.main-container.dock-dragging``, every service iframe ``pointer-events:
none``) and that it is down again afterwards. A sash resize is a real
Playwright pointer gesture.

How a start state is made. Services are opened from the sidebar as a person
would; the exact start arrangement (A | B | C at 50/30/20, a two-tab tile,
...) is then put in the layout holder with ``restore`` — the same entry a
saved dock takes on reload — and every service tab is shown once so its iframe
exists. After every gesture the test waits for the holder to be idle and the
DOM to show exactly the holder's tiles and tabs.

What is read back. The holder's layout as a shape (``row[a,[*b|c],terminal]``:
a tile with one tab is its key, a tile with several lists them with ``*``
before the showing one), and each tile's on-screen rectangle. The drop
highlight is read during the drag (its kind class and rectangle) and compared
with the resulting tile's rectangle after the drop (within 2 px).

The open rule is pinned at the end: a sidebar row click and the agent's open
land as a tab in front of the service tile used last, and a reload brings the
stored arrangement back whole.

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

#: The drop matrix's start: a two-tab tile, a one-tab tile and the terminal's tile.
MATRIX_START = _split(
    "row",
    _tile("t1", "a", "b", showing="a", size=0.4),
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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
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

#: The holder's layout and, per tile, what its header shows: the showing
#: panel's bar (the hidden siblings' bars are not on screen), its tabs from
#: the strip's items (or the lone title), and their rects.
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
    if (!bar.isConnected || bar.parentElement?.classList.contains('dv-inactive-tab')) continue;
    const group = bar.closest('.dv-groupview');
    if (!group) continue;
    const content = group.querySelector('.dv-content-container');
    const strip = bar.querySelector('.tile-tab-tabs');
    const items = strip && !strip.hidden ? [...bar.querySelectorAll('.tile-tab-item')] : [];
    const title = bar.querySelector('.tile-tab-title');
    const own = bar.getAttribute('data-panel-key');
    const tabEls = items.length ? items : (title && !title.hidden ? [title] : []);
    tiles[bar.dataset.tile] = {
      rect: r(group),
      content: content ? r(content) : null,
      bar: r(bar),
      row: r(bar.closest('.dv-tabs-and-actions-container') ?? bar),
      tabs: items.length ? items.map((t) => t.getAttribute('data-panel-key')) : [own],
      tabRects: tabEls.map(r),
      showing: items.length
        ? (bar.querySelector('.tile-tab-item--showing')?.getAttribute('data-panel-key') ?? null)
        : own,
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

#: Put a layout in the holder, then show every service tab once (the hidden
#: ones first, the showing one last) so each has its iframe and the showing
#: ones are visible, as they would be after a person opened them.
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
  for (const tile of tiles) {
    const order = [...tile.tabs.filter((k) => k !== tile.showing), tile.showing];
    for (const key of order) if (key !== 'terminal') focusPanel(key);
  }
  return 'ok';
}
"""

#: One drag, dispatched in the page with a real DataTransfer. The source's own
#: dragstart raises the shield; nothing here touches it. A `tab` source is a
#: strip item; a `tile` source is the tile's header (dockview's own tab drag,
#: which the bridge reads as the tile); a `rail` source is the sidebar row.
_DRAG_JS = r"""
async ({ source, points, drop, leave }) => {
  const sleep = (t) => new Promise((r) => setTimeout(r, t));
  const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
  const root = document.querySelector('#dock-root');
  const barOf = (tile) => [...root.querySelectorAll(`.tile-tab[data-tile="${CSS.escape(tile)}"]`)]
    .find((b) => !b.parentElement?.classList.contains('dv-inactive-tab')) ?? null;
  let el = null;
  const shown = (sel) => [...root.querySelectorAll(sel)]
    .find((n) => !n.closest('.dv-tab')?.classList.contains('dv-inactive-tab')) ?? null;
  if (source.tab) {
    // A lone tab has no strip item: its header is the tab, dragged as the tile.
    el = shown(`.tile-tab-item[data-panel-key="${CSS.escape(source.tab)}"]`)
      ?? shown(`.tile-tab[data-panel-key="${CSS.escape(source.tab)}"]`)?.closest('.dv-tab') ?? null;
  }
  else if (source.tile) el = barOf(source.tile)?.closest('.dv-tab') ?? null;
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
      const marker = document.querySelector('.dock-drop-preview__marker');
      preview = {
        kind,
        rect: { x: b.x, y: b.y, width: b.width, height: b.height },
        look: cs ? [cs.borderTopStyle, cs.borderTopWidth, cs.top, cs.left].join(' ') : null,
        marker: marker ? (() => { const m = marker.getBoundingClientRect(); return { x: m.x, y: m.y, height: m.height }; })() : null,
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
    """Open each service from its sidebar row, as a person does, so it has an iframe."""
    for letter in letters:
        key = S(letter)
        rail_entry(page, key, enabled_only=True).click()
        expect(_iframe(page, key)).to_be_visible(timeout=10_000)
    _settle(page)


# --- geometry -----------------------------------------------------------------


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


def _tab_gap(state: dict, key: str, index: int) -> dict:
    """A point on `key`'s tile header that inserts at `index` among its tabs."""
    tile = state["tiles"][_tile_of(state, key)]
    bar, tabs = tile["bar"], tile["tabRects"]
    y = bar["y"] + bar["height"] / 2
    if not tabs:
        # The agent pane's header: no tabs, so any point on the bar.
        return {"x": bar["x"] + 24, "y": y}
    if index < len(tabs):
        return {"x": tabs[index]["x"] + 2, "y": y}
    last = tabs[-1]
    return {"x": min(last["x"] + last["width"] + 6, bar["x"] + bar["width"] - 2), "y": y}


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


def _item(page: Page, key: str):
    """The strip item of `key` on the bar on screen (a hidden sibling's bar has one too)."""
    return page.locator(
        f'#dock-root .dv-tab:not(.dv-inactive-tab) .tile-tab-item[data-panel-key="{key}"]'
    )


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

        # T2: D onto B's centre: a sidebar row makes a new tile, never a tab —
        # no highlight, nothing changes.
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

#: (source, target, expected shape) from MATRIX_START: row[[*a|b], c, terminal].
#: Targets are on C's tile — a service tile, which takes a tab from a tab or a
#: tile and never from a sidebar row (None: no target) — or beside it: the
#: split line just left of the C|terminal line, and the dock side is the dock's
#: bottom under C; both are edges of C.
_MATRIX = [
    ("tab", "centre", "row[a,[c|*b],terminal]"),
    ("tab", "tabbar", "row[a,[*b|c],terminal]"),
    ("tab", "edge", "row[a,column[c,b],terminal]"),
    ("tab", "split line", "row[a,c,b,terminal]"),
    ("tab", "dock side", "column[row[a,c,terminal],b]"),
    ("tile", "centre", "row[[c|*a|b],terminal]"),
    ("tile", "tabbar", "row[[*a|b|c],terminal]"),
    ("tile", "edge", "row[column[c,[*a|b]],terminal]"),
    ("tile", "split line", "row[c,[*a|b],terminal]"),
    ("tile", "dock side", "column[row[c,terminal],[*a|b]]"),
    ("rail", "centre", None),
    ("rail", "tabbar", None),
    ("rail", "edge", "row[[*a|b],column[c,d],terminal]"),
    ("rail", "split line", "row[[*a|b],c,d,terminal]"),
    ("rail", "dock side", "column[row[[*a|b],c,terminal],d]"),
]
_KIND = {
    "centre": "tab",
    "tabbar": "tab",
    "edge": "split",
    "split line": "split",
    "dock side": "split",
}
#: The placeholder every highlight paints: a 2 px solid stroke, inset 4 px.
_PLACEHOLDER = "solid 2px 4px 4px"


def test_every_drop_type_for_every_source(tmp_path, chromium_browser):
    """Centre, tab bar, edge, split line and dock side, for a tab, a tile and a sidebar row.

    The highlight outlines the result within 2 px (a moved tile, whose own
    space goes to a neighbour first, can land wider), every highlight is the same
    placeholder, none stays painted after the drop, the shield goes down, and a
    moved tile keeps its tabs and showing tab. A row dropped on a header or a
    centre shows nothing and changes nothing. A self-drop shows nothing and
    changes nothing; a drag that leaves the dock leaves no highlight behind.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        looks: set[str] = set()
        for source_kind, target, want in _MATRIX:
            start = _restore(page, MATRIX_START)
            assert start["shape"] == S("row[[*a|b],c,terminal]")
            source = {"tab": {"tab": b}, "tile": {"tile": "t1"}, "rail": {"rail": d}}[source_kind]
            moved = {"tab": b, "tile": a, "rail": d}[source_kind]
            if target == "centre":
                point = _centre(start, c)
            elif target == "tabbar":
                point = _tab_gap(start, c, 0)
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
            drag, after = _drop(page, source, point, kind=_KIND[target])
            assert after["shape"] == S(want), f"{case}: {after['shape']} != {S(want)}"
            # A moved tile's own space goes to a neighbour first, so the result
            # can be wider than the highlight; a tab's and a row's match it.
            if source_kind != "tile":
                assert _close(drag["preview"]["rect"], _rect(after, moved)), (
                    f"{case}: highlight {drag['preview']['rect']} != result {_rect(after, moved)}"
                )
            if target == "tabbar":
                marker = drag["preview"]["marker"]
                assert marker is not None, f"{case}: no insertion marker"
                first = start["tiles"][_tile_of(start, c)]["tabRects"][0]
                assert abs(marker["x"] - first["x"]) <= PX, (case, marker, first)
            if source_kind == "tile":
                moved_tile = after["tiles"][_tile_of(after, a)]
                assert moved_tile["showing"] == a, f"{case}: the moved tile changed its showing tab"
            looks.add(drag["preview"]["look"])
            assert after["previews"] == 0 and not after["shielded"]

        # Every highlight is the one placeholder.
        assert looks == {_PLACEHOLDER}, looks

        # A self-drop (a tile on its own centre) shows nothing and changes nothing.
        start = _restore(page, MATRIX_START)
        drag = _drag(page, {"tile": "t2"}, [_centre(start, c)])
        assert drag["preview"] is None, drag
        after = _settle(page)
        assert after["shape"] == start["shape"]
        _assert_same_rects(start, after, [a, c, TERMINAL])

        # A drag that leaves the dock leaves no highlight, and a cancelled drag changes nothing.
        drag = _drag(page, {"tab": b}, [_centre(start, c)], drop=False, leave=True)
        assert drag["previewAfterLeave"] == 0, "a highlight stayed painted after leaving the dock"
        after = _settle(page)
        assert after["shape"] == start["shape"] and after["previews"] == 0
        assert hub.errors == []


def test_the_agent_pane_is_always_its_own_tile(tmp_path, chromium_browser):
    """Nothing tabs into the agent pane's tile and it never tabs into another; its edges split."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(page, MATRIX_START)

        # Onto the agent pane: a tab, a tile and a row have no target on its
        # header or centre.
        for source in ({"tab": b}, {"tile": "t2"}, {"rail": d}):
            _refused(page, source, _centre(start, TERMINAL), start)
            _refused(page, source, _tab_gap(start, TERMINAL, 0), start)
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
        _refused(page, {"tile": agent_tile}, _tab_gap(start, c, 0), start)
        _, after = _drop(page, {"tile": agent_tile}, _edge(start, c, "bottom"), kind="split")
        assert after["shape"] == S("row[[*a|b],column[c,terminal]]")
        _assert_terminal_shown(page, after)
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Tab bars, the strip below a top bar, sashes, the clamp
# ---------------------------------------------------------------------------


def test_tab_bar_position_and_just_below_a_top_bar_is_the_dock_top(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        three = _split(
            "row",
            _tile("t1", "a", "b", showing="a", size=0.6),
            _tile("t2", "c", size=0.2),
            _tile("t3", TERMINAL, size=0.2),
        )

        # A tab dropped between two tabs lands there and is shown.
        start = _restore(page, three)
        drag, after = _drop(page, {"tab": c}, _tab_gap(start, a, 1), kind="tab")
        assert after["shape"] == S("row[[a|*c|b],terminal]")
        tabs = start["tiles"]["t1"]["tabRects"]
        assert abs(drag["preview"]["marker"]["x"] - tabs[1]["x"]) <= PX, drag["preview"]

        # Reordering within the bar: b to the front.
        start = _restore(page, three)
        _, after = _drop(page, {"tab": b}, _tab_gap(start, a, 0), kind="tab")
        assert after["shape"] == S("row[[*b|a],c,terminal]")

        # A tile's top edge, below the dock's top strip: its top half.
        start = _restore(page, three)
        drag, after = _drop(page, {"rail": d}, _edge(start, a, "top"), kind="split")
        assert after["shape"] == S("row[column[d,[*a|b]],c,terminal]")
        rd = _rect(after, d)
        assert _close(rd, _half(_rect(start, a), "top"), 2 * PX), (rd, _rect(start, a))
        assert _close(drag["preview"]["rect"], rd)

        # Just below a top-row bar is the dock's top strip: a row across the dock.
        start = _restore(page, three)
        drag, after = _drop(page, {"rail": d}, _below_bar(start, a), kind="split")
        assert after["shape"] == S("column[d,row[[*a|b],c,terminal]]")
        assert _close(drag["preview"]["rect"], _rect(after, d))

        # The same below the terminal's bar.
        start = _restore(page, three)
        _, after = _drop(page, {"tab": b}, _below_bar(start, TERMINAL), kind="split")
        assert after["shape"] == S("column[b,row[a,c,terminal]]")
        assert hub.errors == []


def test_the_whole_header_row_takes_a_tab(tmp_path, chromium_browser):
    """Right of the last tab, over the header's free space and actions, a drop adds a tab."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, MATRIX_START)
        row = start["tiles"]["t1"]["row"]
        last = start["tiles"]["t1"]["tabRects"][-1]
        assert row["x"] + row["width"] - 8 > last["x"] + last["width"] + 40, (row, last)
        point = {"x": row["x"] + row["width"] - 8, "y": row["y"] + row["height"] / 2}
        _, after = _drop(page, {"tab": c}, point, kind="tab")
        assert after["shape"] == S("row[[a|b|*c],terminal]")
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


def test_a_tile_too_narrow_to_halve_takes_a_tab(tmp_path, chromium_browser):
    """B is 60 px wide: halves of 30 px are under 40, so its sides are its centre.

    The drop adds C as a tab of B, and nothing else moves.
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
        drag, after = _drop(page, {"tile": "t3"}, point, kind="tab")
        assert after["shape"] == S("row[a,[b|*c]]")
        _assert_same_rects(start, after, [a])
        assert _close(drag["preview"]["rect"], _rect(start, b)), (drag["preview"], rb)
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Input after drags and tab switches
# ---------------------------------------------------------------------------


def test_services_and_terminal_take_input_after_a_drag_and_a_tab_switch(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, MATRIX_START)

        _drop(page, {"tile": _tile_of(start, c)}, _edge(start, a, "bottom"), kind="split")
        _type_in_service(page, a, "after-drag-a")
        _type_in_service(page, c, "after-drag-c")
        _type_in_terminal(page, "afterdrag1")

        # A tab hidden behind another, shown again by a click on its item.
        _item(page, b).click()
        state = _settle(page)
        assert state["tiles"][_tile_of(state, b)]["showing"] == b
        _type_in_service(page, b, "shown-b")
        _item(page, a).click()
        _settle(page)
        expect(_iframe(page, b)).to_be_hidden()
        _item(page, b).click()
        _settle(page)
        _type_in_service(page, b, "shown-again-b")
        _type_in_terminal(page, "afterswitch2")
        assert hub.errors == []


def test_a_tab_click_then_a_drop_elsewhere_keeps_that_tab_showing(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, MATRIX_START)
        _item(page, b).click()
        shown = _settle(page)
        assert shown["shape"] == S("row[[a|*b],c,terminal]")
        _drop(page, {"tile": _tile_of(start, c)}, _edge(start, TERMINAL, "bottom"), kind="split")
        after = _settle(page)
        assert after["shape"] == S("row[[a|*b],column[terminal,c]]")
        expect(_iframe(page, b)).to_be_visible()
        expect(_iframe(page, a)).to_be_hidden()
        _type_in_service(page, b, "still-b")
        assert hub.errors == []


def test_the_strip_moves_the_showing_tab_from_the_keyboard(tmp_path, chromium_browser):
    """Arrow keys on the focused strip switch tabs; Delete closes the one in front."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        _restore(page, MATRIX_START)
        strip = page.locator('#dock-root .tile-tab[data-tile="t1"]:visible .tile-tab-tabs')
        strip.focus()
        page.keyboard.press("ArrowRight")
        state = _settle(page)
        assert state["shape"] == S("row[[a|*b],c,terminal]")
        expect(_iframe(page, b)).to_be_visible()
        # The strip of the bar now showing keeps the keyboard.
        expect(
            page.locator('#dock-root .tile-tab[data-tile="t1"]:visible .tile-tab-tabs')
        ).to_be_focused()
        page.keyboard.press("Delete")
        state = _settle(page)
        assert state["shape"] == S("row[a,c,terminal]")
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Close and the way back, the split-from neighbour
# ---------------------------------------------------------------------------


def test_close_and_the_way_back_through_the_sidebar_row(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")
        start = _restore(page, MATRIX_START)

        def close_showing(tile: str) -> dict:
            bar = page.locator(f'#dock-root .tile-tab[data-tile="{tile}"]:visible')
            bar.hover()
            bar.locator(".tile-tab-close").click()
            return _settle(page)

        # A tab: shown, closed, then back from its sidebar row.
        _item(page, b).click()
        _settle(page)
        state = close_showing("t1")
        assert state["shape"] == S("row[a,c,terminal]")
        _assert_same_rects(start, state, [a, c, TERMINAL])
        rail_entry(page, b, enabled_only=True).click()
        state = _settle(page)
        assert b in [k for t in state["held"] for k in t["tabs"]], state["shape"]
        _type_in_service(page, b, "back-b")

        # A tile (its last tab): one neighbour grows, then back from the sidebar.
        before = _restore(page, MATRIX_START)
        state = close_showing("t2")
        assert state["shape"] == S("row[[*a|b],terminal]")
        grew = [k for k in (a, TERMINAL) if not _close(_rect(before, k), _rect(state, k))]
        assert len(grew) == 1, (before["tiles"], state["tiles"])
        rail_entry(page, c, enabled_only=True).click()
        state = _settle(page)
        assert c in [k for t in state["held"] for k in t["tabs"]], state["shape"]
        _type_in_service(page, c, "back-c")

        # The agent pane: closed, then back from its sidebar row, and it takes keys.
        _restore(page, MATRIX_START)
        state = close_showing("t3")
        assert TERMINAL not in [k for t in state["held"] for k in t["tabs"]], state["shape"]
        rail_entry(page, TERMINAL).click()
        state = _settle(page)
        assert TERMINAL in [k for t in state["held"] for k in t["tabs"]], state["shape"]
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


def test_an_open_lands_as_a_tab_in_front_of_the_service_tile_used_last(tmp_path, chromium_browser):
    """A row click and the agent's open add a tab in front; nothing resizes."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        start = _restore(page, MATRIX_START)

        # C's tile is the one used last: D's row lands D in front of C there.
        page.locator('#dock-root .tile-tab[data-tile="t2"]:visible .tile-tab-title').click()
        _settle(page)
        rail_entry(page, d, enabled_only=True).click()
        after = _settle(page)
        assert after["shape"] == S("row[[*a|b],[c|*d],terminal]")
        _assert_same_rects(start, after, [a, c, TERMINAL])
        expect(_iframe(page, d)).to_be_visible(timeout=5_000)
        expect(_iframe(page, c)).to_be_hidden(timeout=5_000)
        _type_in_service(page, d, "tab-d")

        # A covered panel's row brings it forward where it is.
        rail_entry(page, b, enabled_only=True).click()
        after = _settle(page)
        assert after["shape"] == S("row[[a|*b],[c|*d],terminal]")
        _assert_same_rects(start, after, [a, c, TERMINAL])

        # The agent's open lands the same way as the row click.
        start = _restore(page, MATRIX_START)
        r = requests.post(
            f"{hub.base_url}/api/panel-op/open",
            json={"request_id": "tiles-open-1", "member": d},
            timeout=10,
        )
        assert r.status_code == 200, r.text
        expect(_item(page, d)).to_have_count(1, timeout=10_000)
        after = _settle(page)
        assert after["shape"] == S("row[[*a|b],[c|*d],terminal]")
        _assert_same_rects(start, after, [a, c, TERMINAL])
        _type_in_service(page, d, "agent-d")
        assert hub.errors == []


def test_a_reload_brings_the_stored_layout_back_whole(tmp_path, chromium_browser):
    """Tiles, tabs, the tab in front and the sizes come back as they were saved."""
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
        for key in (a, c):
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
                    for key in (a, c, TERMINAL)
                )
            ),
        )
        assert after["shape"] == S("row[[*a|b],c,terminal]") == before["shape"]
        for key in (a, c, TERMINAL):
            assert _close(_rect(before, key), _rect(after, key), 2 * PX), (
                key,
                _rect(before, key),
                _rect(after, key),
            )
        assert b in after["tiles"][_tile_of(after, a)]["tabs"]
        _type_in_service(page, a, "after-reload")
        assert hub.errors == []
