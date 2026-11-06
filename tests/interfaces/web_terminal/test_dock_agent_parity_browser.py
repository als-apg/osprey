"""The MCP panel tools, run against a live OSPREY web terminal: an agent's dock writes.

Every write here is made by calling the real MCP tool function
(``drop_panel``, ``close_panel``, ``open_panel``, ``open_panel_set``,
``arrange_workspace``; ARIEL's and Phoebus's own open paths) with the tool's
HTTP base pointed at a live web terminal. The tool posts to
``/api/panel-op/<op>``, the server relays one ``panel_op`` frame to the
browser that holds the dock, that browser runs it on its layout holder and
posts the outcome back; the tool returns the resulting layout and ``rev`` or
raises the named refusal. Nothing in between is stubbed except Phoebus's own
bridge (there is no Phoebus product here; its web-terminal open is real).

What is proven, against what a person does in the same dock:

- **Parity.** Every pointer zone that splits (a tile's edge, next to a split
  line, the dock's side) for every source (an open panel, a whole tile, a
  panel not open yet), plus closing a panel, gives the layout a person's drag
  (or the tile's ×) gives from the same start: same tree, panels and sizes,
  and the same on-screen rectangles. The person's drags are real HTML5 drags
  dispatched from the real source element (its own ``dragstart`` raises the
  drag shield). A drop into a tile (its centre or its header) shows a person
  nothing and is refused ``one_panel_per_tile`` for the agent, both changing
  nothing.
- **Sets.** A set of three opens side by side, and in a given layout of one
  panel per tile, one call each; after each the terminal keeps its tile id,
  its rectangle and its panel. A layout putting two panels in one tile is
  refused ``one_panel_per_tile``.
- **The operator keeps the terminal.** An agent open (with and without
  ``url``) landing while the operator types never covers the terminal, and the
  typed keys reach the shell (read from the shell's side: a stand-in program
  appends everything it reads to a log file).
- **The terminal is not agent-movable.** As a drop source (panel or tile), a
  close target or an open member it is refused ``terminal_not_agent_movable``
  and nothing changes.
- **Revisions.** Two drops in a row, the second naming the tile and ``rev``
  the first returned, both succeed; an op carrying the pre-drag ``base_rev``
  while a person's drag is still unacknowledged is refused ``stale_base``; an
  op that arrives mid-drag is held off the screen until the drag ends, then
  applies. A person who undoes an agent's open (back to the exact pre-op
  layout) is reported, so the agent never reads a dock that is no longer on
  screen.
- **Which browser tab answers.** Closing the reporting browser tab moves the
  next write to the other open tab; with none open, ``no_dock_client``.
- **ARIEL and Phoebus** open their panels through the same op path, one dock
  tile each time, in the one browser tab that holds the dock.

Run:

    uv run pytest tests/interfaces/web_terminal/test_dock_agent_parity_browser.py -rs
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.interfaces._serving import run_app_server
from tests.interfaces._browser import (
    open_hub_page,
    rail_entry,
    wait_for_dock_settled,
)
from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.web_terminal._surface import arm_terminal_tab
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover - the chromium fixture skips first
    Page = object  # type: ignore[assignment,misc]

pytestmark = [pytest.mark.browser, pytest.mark.slow]

VIEWPORT = {"width": 1600, "height": 1000}
#: The five generic services, then the two whose own MCP servers open them.
KEYS = ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e"]
ARIEL = "ariel"
PHOEBUS = "phoebus"
TERMINAL = "terminal"
#: Rect tolerance between the person's result and the agent's.
PX = 2
#: The engine's STRIP_PX: the dock's outer drop strips, this deep on each side.
STRIP_PX = 48
#: The engine's EDGE_RATIO: a tile's edge band, as a share of its size.
EDGE_RATIO = 0.31
#: Size (fraction) tolerance between the person's result and the agent's.
SIZE_TOL = 0.01
_SETTLE_S = 8.0

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def S(text: str) -> str:
    """A shape written with one-letter service names: ``a`` is ``svc-a``."""
    return re.sub(r"(?<![\w-])([a-e])(?![\w-])", r"svc-\1", text)


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


#: A over B, C, and the terminal's own tile: C's and the terminal's tiles sit
#: where they did when B was a tab behind A.
MATRIX_START = _split(
    "row",
    _split("column", _tile("t1", "a", size=0.5), _tile("t4", "b", size=0.5), size=0.4),
    _tile("t2", "c", size=0.3),
    _tile("t3", TERMINAL, size=0.3),
)

#: One service tile and the terminal's tile.
ONE_SERVICE = _split("row", _tile("t1", "a", size=0.6), _tile("t3", TERMINAL, size=0.4))

# ---------------------------------------------------------------------------
# The live services, the shell stand-in and the hub
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


#: The PTY's program: every byte the terminal sends is appended to argv[1].
_KEYS_SHELL = (
    "import os, sys\n"
    "out = open(sys.argv[1], 'ab', buffering=0)\n"
    "while True:\n"
    "    chunk = os.read(0, 4096)\n"
    "    if not chunk:\n"
    "        break\n"
    "    out.write(chunk)\n"
)


@dataclass
class Hub:
    base_url: str
    app: object
    keys_log: Path
    errors: list[str] = field(default_factory=list)


@contextmanager
def _hub(tmp_path: Path) -> Iterator[Hub]:
    """A live web terminal with the services as custom panels.

    The MCP side's web-terminal base is pointed at this server, so every tool
    call below reaches it exactly as a deployed agent's would.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    (workspace / "seed_artifact.txt").write_text("seed\n")
    keys_log = tmp_path / "shell-keys.log"
    keys_log.write_bytes(b"")
    all_keys = [*KEYS, ARIEL, PHOEBUS]
    with ExitStack() as stack:
        urls = [stack.enter_context(_service(key)) for key in all_keys]
        custom = [
            {"id": key, "label": key.upper(), "url": url, "healthEndpoint": None, "path": "/"}
            for key, url in zip(all_keys, urls, strict=True)
        ]
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace)},
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=({"artifacts"}, custom, None),
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._launch_panel_server",
                side_effect=publish_artifact_url(),
            )
        )
        stack.enter_context(
            patch(
                "osprey_connectors.workspace.resolve_shared_data_root",
                return_value=tmp_path / "agent_data",
            )
        )
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=[sys.executable, "-c", _KEYS_SHELL, str(keys_log)])
        base_url = stack.enter_context(run_app_server(app))
        app.state.project_cwd = str(tmp_path)
        app.state.visible_panels = ["artifacts", *all_keys]
        # The agent side: panel_op and fetch_panels resolve the base through this.
        stack.enter_context(patch("osprey.mcp_server.http.web_terminal_url", return_value=base_url))
        yield Hub(base_url, app, keys_log)


def _open(browser, hub: Hub, *, terminal: bool = False) -> Page:
    """Open the hub; with ``terminal`` the Agent pane boots on its terminal surface."""
    if terminal:
        page = browser.new_page(viewport=VIEWPORT)
        arm_terminal_tab(page)
        page.goto(hub.base_url, wait_until="domcontentloaded")
        expect(rail_entry(page, "terminal", enabled_only=True)).to_be_attached(timeout=15_000)
        expect(page.locator(".dv-groupview").first).to_be_visible(timeout=15_000)
    else:
        page = open_hub_page(browser, hub.base_url, viewport=VIEWPORT)
    page.on("pageerror", lambda e: hub.errors.append(str(e)))
    wait_for_dock_settled(page)
    return page


# ---------------------------------------------------------------------------
# The agent: the real MCP tool functions
# ---------------------------------------------------------------------------


def _tool(name: str, **kwargs) -> dict:
    """Call one panel tool as the MCP server would, and parse its JSON answer."""
    from osprey.mcp_server.workspace.tools import panel_tools

    raw = _run_async(get_tool_fn(getattr(panel_tools, name)), **kwargs)
    return json.loads(raw)


def _run_async(fn, *args, **kwargs):
    """Run an async MCP tool function to completion on its own thread.

    Playwright's sync API keeps an event loop running on the test's thread, so
    ``asyncio.run`` cannot be called there; a tool gets a fresh loop instead,
    as it would inside the MCP server.
    """
    out: dict = {}

    def run():
        try:
            out["result"] = asyncio.run(fn(*args, **kwargs))
        except BaseException as exc:  # re-raised on the test's thread
            out["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), f"{fn.__name__} never returned"
    if "error" in out:
        raise out["error"]
    return out["result"]


def _refused(reason: str, name: str, **kwargs) -> dict:
    """Call a panel tool that must be refused ``reason``; return the error envelope."""
    with assert_raises_error(error_type=reason) as ctx:
        _tool(name, **kwargs)
    envelope = ctx["envelope"]
    assert envelope["details"]["reason"] == reason, envelope
    return envelope


class _Background:
    """One tool call on a worker thread (Playwright's sync API stays on this one)."""

    def __init__(self, fn, *args, **kwargs):
        self.result = None
        self.error: BaseException | None = None

        def run():
            try:
                self.result = fn(*args, **kwargs)
            except BaseException as exc:  # re-raised in join()
                self.error = exc

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def join(self, timeout: float = 15.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the tool call never returned"
        if self.error is not None:
            raise self.error
        return self.result


# ---------------------------------------------------------------------------
# In-page reads
# ---------------------------------------------------------------------------

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

_RESTORE_JS = """
async (layout) => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const holder = getLayoutHolder();
  if (!holder) return 'no holder';
  return holder.restore(layout) ? 'ok' : 'refused';
}
"""

#: The holder's acknowledgement state and this page's dock client id.
_HOLDER_JS = """
async () => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const { getPageClientId } = await import('/static/js/api.js');
  const holder = getLayoutHolder();
  return { dirty: !!holder?.dirty, acked: holder?.ackedRev ?? null, client: getPageClientId() };
}
"""

_DRAG_PRELUDE = r"""
  const sleep = (t) => new Promise((r) => setTimeout(r, t));
  const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
  const root = document.querySelector('#dock-root');
  const find = (source) => {
    // A panel's bar is its tile's header: dockview's header drag, read as the tile.
    if (source.tab) return root.querySelector(`.tile-tab[data-panel-key="${CSS.escape(source.tab)}"]`)?.closest('.dv-tab') ?? null;
    if (source.grip) return root.querySelector(`.tile-tab[data-tile="${CSS.escape(source.grip)}"]`)?.closest('.dv-tab') ?? null;
    if (source.rail) return document.querySelector(`button.panel-rail-button[data-panel-id="${CSS.escape(source.rail)}"]`);
    return null;
  };
  const shieldUp = () => !!document.querySelector('.main-container.dock-dragging');
"""

#: Start a drag from the real source element and hover it over `point`; the drag
#: stays live (window.__agentParityDrag) until _END_DRAG_JS drops it.
_BEGIN_DRAG_JS = (
    "async ({ source, point }) => {"
    + _DRAG_PRELUDE
    + r"""
  const el = find(source);
  if (!el) return { error: `no source element for ${JSON.stringify(source)}` };
  const dataTransfer = new DataTransfer();
  const init = (x, y) => ({ bubbles: true, cancelable: true, composed: true, dataTransfer, clientX: x, clientY: y });
  const s = el.getBoundingClientRect();
  el.dispatchEvent(new DragEvent('dragstart', init(s.x + s.width / 2, s.y + s.height / 2)));
  let shieldRose = false;
  for (let i = 0; i < 60 && !shieldRose; i += 1) {
    if (shieldUp()) shieldRose = true; else await sleep(16);
  }
  const target = document.elementFromPoint(point.x, point.y);
  if (!target) return { error: 'nothing at the target point' };
  target.dispatchEvent(new DragEvent('dragenter', init(point.x, point.y)));
  for (let i = 0; i < 2; i += 1) {
    target.dispatchEvent(new DragEvent('dragover', init(point.x, point.y)));
    await frame();
  }
  window.__agentParityDrag = { el, target, point, init };
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  return {
    shieldRose,
    dragLive: !!getLayoutHolder()?.dragLive,
    preview: document.querySelector('.dock-drop-preview__rect')?.className ?? null,
  };
}
"""
)

_END_DRAG_JS = """
async () => {
  const drag = window.__agentParityDrag;
  if (!drag) return 'no drag';
  const { el, target, point, init } = drag;
  target.dispatchEvent(new DragEvent('dragover', init(point.x, point.y)));
  target.dispatchEvent(new DragEvent('drop', init(point.x, point.y)));
  (el.isConnected ? el : document).dispatchEvent(new DragEvent('dragend', init(point.x, point.y)));
  delete window.__agentParityDrag;
  return 'ok';
}
"""

#: One complete drag (dragstart on the real source, hover, drop, dragend).
_DRAG_JS = (
    "async ({ source, point }) => {"
    + _DRAG_PRELUDE
    + r"""
  const el = find(source);
  if (!el) return { error: `no source element for ${JSON.stringify(source)}` };
  const dataTransfer = new DataTransfer();
  const init = (x, y) => ({ bubbles: true, cancelable: true, composed: true, dataTransfer, clientX: x, clientY: y });
  const s = el.getBoundingClientRect();
  el.dispatchEvent(new DragEvent('dragstart', init(s.x + s.width / 2, s.y + s.height / 2)));
  let shieldRose = false;
  for (let i = 0; i < 60 && !shieldRose; i += 1) {
    if (shieldUp()) shieldRose = true; else await sleep(16);
  }
  const iframes = [...document.querySelectorAll('.dock-iframe-overlay iframe')];
  const iframesShielded = iframes.every((f) => f.style.pointerEvents === 'none');
  const target = document.elementFromPoint(point.x, point.y);
  if (!target) return { error: 'nothing at the target point' };
  target.dispatchEvent(new DragEvent('dragenter', init(point.x, point.y)));
  for (let i = 0; i < 2; i += 1) {
    target.dispatchEvent(new DragEvent('dragover', init(point.x, point.y)));
    await frame();
  }
  const box = document.querySelector('.dock-drop-preview__rect');
  const kind = box
    ? [...box.classList].find((c) => c.startsWith('dock-drop-preview__rect--'))?.slice(25) ?? null
    : null;
  const hit = { sash: !!target.closest('.dv-sash'), inDock: root.contains(target) };
  target.dispatchEvent(new DragEvent('drop', init(point.x, point.y)));
  (el.isConnected ? el : document).dispatchEvent(new DragEvent('dragend', init(point.x, point.y)));
  return { shieldRose, iframesShielded, kind, hit };
}
"""
)


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


def _quiet(page: Page, hub: Hub) -> int:
    """Wait until the server holds this page's layout as current and the page knows it.

    The page is the dock client, its holder is not dirty, and its acked rev is
    the server's rev: an agent reading the panels now reads this layout.
    Returns that rev.
    """
    deadline = time.monotonic() + _SETTLE_S
    seen = None
    while time.monotonic() < deadline:
        holder = page.evaluate(_HOLDER_JS)
        state = hub.app.state
        rev = getattr(state, "dock_layout_rev", 0)
        seen = (holder, rev, getattr(state, "dock_client_id", None))
        if (
            not holder["dirty"]
            and holder["acked"] == rev
            and getattr(state, "dock_client_id", None) == holder["client"]
        ):
            diffs = _differences(_canon(state.dock_layout), _canon(_state(page)["layout"]))
            assert not diffs, (
                f"the page counts rev {rev} as acked, but the server holds another layout: {diffs}"
            )
            return rev
        page.wait_for_timeout(100)
    raise AssertionError(f"the server never acknowledged the page's layout: {seen}")


def _bounced(layout: dict) -> dict:
    """`layout` with every tile id renamed: the same screen, a different report."""
    if "tile" in layout:
        return {**layout, "tile": layout["tile"] + "x"}
    return {**layout, "items": [_bounced(item) for item in layout["items"]]}


def _restore(page: Page, hub: Hub, layout: dict) -> dict:
    """Put `layout` in the holder, wait for the dock and the server to hold it.

    The layout is first restored with renamed tile ids, so restoring it is
    always a change the page reports — even when it equals the page's last
    dock-sync report and an agent op's result has moved the server on since
    (the defect pinned by test_a_person_undoing_an_agent_open_is_reported).
    """
    assert page.evaluate(_RESTORE_JS, _bounced(layout)) == "ok"
    _settle(page)
    _quiet(page, hub)
    assert page.evaluate(_RESTORE_JS, layout) == "ok"
    state = _settle(page)
    for tile in state["held"]:
        if tile["showing"] != TERMINAL:
            expect(_iframe(page, tile["showing"])).to_be_visible(timeout=5_000)
    state = _settle(page)
    _quiet(page, hub)
    return state


def _drag(page: Page, source: dict, point: dict) -> dict:
    result = page.evaluate(_DRAG_JS, {"source": source, "point": point})
    assert "error" not in result, result
    # The source's own dragstart raised the shield, over every service iframe.
    assert result["shieldRose"], f"the drag of {source} raised no shield"
    assert result["iframesShielded"], f"the drag of {source} left an iframe unshielded"
    return result


def _iframe(page: Page, key: str):
    return page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{key}"]')


def _bar(page: Page, key: str):
    """The header bar of the tile showing ``key``."""
    return page.locator(f'#dock-root .tile-tab[data-panel-key="{key}"]')


def _rail_open(page: Page, key: str) -> None:
    """Open a panel from its sidebar row — it replaces the panel of the service
    tile used last — and wait for its bar."""
    rail_entry(page, key, enabled_only=True).click()
    expect(_bar(page, key)).to_have_count(1, timeout=5_000)


def _open_services(page: Page, *keys: str) -> None:
    for key in keys:
        _rail_open(page, S(key) if len(key) == 1 else key)
    _settle(page)


def _close_by_hand(page: Page, key: str) -> dict:
    """Press the × of the tile showing ``key``: the tile closes."""
    bar = _bar(page, key)
    bar.hover()
    bar.locator(".tile-tab-close").click()
    expect(_bar(page, key)).to_have_count(0, timeout=5_000)
    return _settle(page)


def _focus_by_hand(page: Page, key: str) -> dict:
    """Click the title of the tile showing ``key``: the person's focus gesture."""
    _bar(page, key).locator(".tile-tab-title").click()
    return _settle(page)


# --- geometry -----------------------------------------------------------------


def _keys(state_or_layout: dict) -> list[str]:
    layout = state_or_layout.get("layout", state_or_layout)
    found: list[str] = []

    def walk(n: dict) -> None:
        if "tile" in n:
            found.extend(n["tabs"])
        else:
            for item in n["items"]:
                walk(item)

    if layout:
        walk(layout)
    return found


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


def _edge_bottom(state: dict, key: str) -> dict:
    """A point in the bottom edge band of `key`'s tile, clear of the dock's bottom strip."""
    c = _content(state, key)
    inset = _edge_inset(c["height"])
    return {"x": c["x"] + c["width"] / 2, "y": c["y"] + c["height"] - inset}


def _on_bar(state: dict, key: str) -> dict:
    """A point on the header row of `key`'s tile, over its title."""
    row = state["tiles"][_tile_of(state, key)]["row"]
    return {"x": row["x"] + 24, "y": row["y"] + row["height"] / 2}


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


def _close(a: dict, b: dict, tol: float = PX) -> bool:
    return all(abs(a[k] - b[k]) <= tol for k in ("x", "y", "width", "height"))


# --- comparing two layouts ------------------------------------------------------


def _canon(layout: dict) -> dict:
    """`layout` with tile ids replaced by their tabs, so two runs compare by content.

    Tile ids are minted per run; a tile is identified here by its tab list,
    which is unique within a layout. ``from`` (the tile a tile split from) is
    mapped the same way, so rule 5's neighbour is compared too.
    """
    ids: dict[str, str] = {}

    def collect(n: dict) -> None:
        if "tile" in n:
            ids[n["tile"]] = "|".join(n["tabs"])
        else:
            for item in n["items"]:
                collect(item)

    collect(layout)

    def walk(n: dict) -> dict:
        if "tile" in n:
            out: dict = {"tabs": list(n["tabs"]), "showing": n["showing"]}
            if n.get("from") is not None:
                out["from"] = ids.get(n["from"], "<closed>")
        else:
            out = {"split": n["split"], "items": [walk(item) for item in n["items"]]}
        out["size"] = n.get("size")
        return out

    return walk(layout)


def _differences(a: object, b: object, at: str = "") -> list[str]:
    """Where two canonical layouts differ; sizes compare within SIZE_TOL."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key == "size" and a.get(key) is not None and b.get(key) is not None:
                if abs(a[key] - b[key]) > SIZE_TOL:
                    out.append(f"{at}.size: {a[key]:.4f} != {b[key]:.4f}")
                continue
            out += _differences(a.get(key), b.get(key), f"{at}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _differences(x, y, f"{at}[{i}]")
        ]
    return [] if a == b else [f"{at}: {a!r} != {b!r}"]


def _assert_same_layout(person: dict, agent: dict, case: str) -> None:
    """The agent's settled dock equals the person's: tree, tabs, sizes, rectangles."""
    diffs = _differences(_canon(person["layout"]), _canon(agent["layout"]))
    assert not diffs, f"{case}: person {person['shape']} vs agent {agent['shape']}: {diffs}"
    for key in _keys(person):
        assert _close(_rect(person, key), _rect(agent, key)), (
            f"{case}: {key} at {_rect(person, key)} by hand, {_rect(agent, key)} by the agent"
        )


def _assert_returned_is_shown(result: dict, state: dict, case: str) -> None:
    """The layout the tool returned is the layout the dock now shows."""
    diffs = _differences(_canon(result["layout"]), _canon(state["layout"]))
    assert not diffs, f"{case}: the tool returned a layout the dock does not show: {diffs}"


def _assert_terminal_untouched(before: dict, after: dict, case: str) -> None:
    """The terminal keeps its tile id, its rectangle and its showing tab."""
    tid = _tile_of(before, TERMINAL)
    assert _tile_of(after, TERMINAL) == tid, f"{case}: the terminal moved tile: {after['shape']}"
    assert after["tiles"][tid]["showing"] == TERMINAL, f"{case}: {after['shape']}"
    assert after["tiles"][tid]["tabs"] == before["tiles"][tid]["tabs"], f"{case}: {after['shape']}"
    assert _close(before["tiles"][tid]["rect"], after["tiles"][tid]["rect"]), (
        f"{case}: the terminal's tile changed size: "
        f"{before['tiles'][tid]['rect']} -> {after['tiles'][tid]['rect']}"
    )


# --- the terminal and its shell -------------------------------------------------


def _focus_terminal(page: Page) -> None:
    target = page.locator("#dock-root .xterm").first
    expect(target).to_be_visible(timeout=10_000)
    target.click(timeout=5_000)
    page.wait_for_function(
        "() => document.activeElement?.classList.contains('xterm-helper-textarea')",
        timeout=5_000,
    )


def _assert_shell_got(hub: Hub, text: str) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if text.encode() in hub.keys_log.read_bytes():
            return
        time.sleep(0.1)
    raise AssertionError(f"{text!r} never reached the shell: {hub.keys_log.read_bytes()[-400:]!r}")


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
# Parity: every drop type x every source, close tab and close tile
# ---------------------------------------------------------------------------

#: (source kind, target kind) from MATRIX_START: row[column[a,b], c, terminal].
#: Every split target for every source; the targets inside a tile (its centre,
#: its header) take nothing from any source and are _ZONES.
_MATRIX = [
    (source, target)
    for source in ("tab", "tile", "new")
    for target in ("edge", "split line", "dock side")
]
_ZONES = [(source, target) for source in ("tab", "tile", "new") for target in ("centre", "tabbar")]


def _person_point(start: dict, target: str) -> dict:
    """Where a person drops for this case (the points test_dock_tiles_browser proves).

    Every target is on C's tile or beside it: the Agent pane's tile takes
    nothing either (dock-drop.js admits no centre or header target on any
    tile), so the places a person can try are a service tile's.
    """
    if target == "centre":
        return _centre(start, S("c"))
    if target == "tabbar":
        return _on_bar(start, S("c"))
    if target == "edge":
        return _edge_bottom(start, S("c"))
    if target == "split line":
        return _near_split(start, S("c"), TERMINAL)
    return _dock_bottom(start, S("c"))


def _agent_target(target: str) -> dict:
    """The same place as a drop target, named by MATRIX_START's tile ids."""
    if target == "centre":
        return {"tile": "t2", "zone": "centre"}
    if target == "tabbar":
        return {"tile": "t2", "zone": "tab", "index": 0}
    if target == "edge":
        return {"tile": "t2", "edge": "bottom"}
    if target == "split line":
        return {"tile": "t2", "edge": "right"}
    return {"outer": "bottom"}


def test_every_agent_drop_and_close_gives_the_persons_layout(tmp_path, chromium_browser):
    """``drop_panel`` and ``close_panel`` match a person, case by case.

    Each case starts twice from the same layout: once a person drags (or
    closes a tile by its ×), once the agent makes the equivalent call with
    the ``rev`` it just read. The two settled docks must have the same tree,
    panels and sizes (tile ids aside, which are minted) and the same
    rectangles; and the layout the tool returned must be the one the dock
    shows, so an agent never has to re-read. A drop into a tile changes
    nothing either way: the person's drag paints nothing, and the agent's is
    refused ``one_panel_per_tile``.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c, d = (S(x) for x in "abcd")
        persons = {"tab": {"tab": b}, "tile": {"grip": "t1"}, "new": {"rail": d}}
        agents = {"tab": {"tab": b}, "tile": {"tile": "t1"}, "new": {"new": {"panel": d}}}

        for source_kind, target in _MATRIX:
            case = f"{source_kind} -> {target}"
            start = _restore(page, hub, MATRIX_START)
            assert start["shape"] == S("row[column[a,b],c,terminal]")
            drag = _drag(page, persons[source_kind], _person_point(start, target))
            assert drag["kind"] is not None, f"{case}: the person's drag showed no highlight"
            person = _settle(page)
            assert person["shape"] != start["shape"], f"{case}: the person's drag changed nothing"

            _restore(page, hub, MATRIX_START)
            rev = _tool("list_panels")["rev"]
            result = _tool(
                "drop_panel",
                source=agents[source_kind],
                target=_agent_target(target),
                base_rev=rev,
            )
            assert result["status"] == "success" and result["rev"] > rev, (case, result)
            agent = _settle(page)
            _assert_returned_is_shown(result, agent, case)
            _assert_same_layout(person, agent, case)

        # Into a tile: nothing for the person, one_panel_per_tile for the agent.
        for source_kind, target in _ZONES:
            case = f"{source_kind} -> {target}"
            start = _restore(page, hub, MATRIX_START)
            drag = _drag(page, persons[source_kind], _person_point(start, target))
            assert drag["kind"] is None, f"{case}: the person's drag showed a highlight"
            assert _settle(page)["shape"] == start["shape"], f"{case}: the person's drop moved"

            start = _restore(page, hub, MATRIX_START)
            rev = _tool("list_panels")["rev"]
            envelope = _refused(
                "one_panel_per_tile",
                "drop_panel",
                source=agents[source_kind],
                target=_agent_target(target),
                base_rev=rev,
            )
            assert "Nothing changed" in envelope["error_message"], case
            assert _settle(page)["shape"] == start["shape"], f"{case}: the refused drop moved"
            assert hub.app.state.dock_layout_rev == rev, f"{case}: the rev moved"

        # Close a panel: the tile showing it closes, by its × or by the agent.
        for key in (b, c):
            case = f"close {key}"
            _restore(page, hub, MATRIX_START)
            person = _close_by_hand(page, key)
            _rail_open(page, key)  # back as a person brings it back, before the next start
            _restore(page, hub, MATRIX_START)
            result = _tool("close_panel", panel_id=key)
            assert result["panel"] == key, result
            agent = _settle(page)
            _assert_returned_is_shown(result, agent, case)
            _assert_same_layout(person, agent, case)
            assert key not in _keys(agent)

        # A panel no tile shows has no tile to close.
        start = _restore(page, hub, MATRIX_START)
        _refused("unknown_key", "close_panel", panel_id=d)
        assert _settle(page)["shape"] == start["shape"]
        assert hub.errors == []


# ---------------------------------------------------------------------------
# A set of three
# ---------------------------------------------------------------------------


def test_a_set_of_three_opens_side_by_side_and_by_layout_leaving_the_terminal(
    tmp_path, chromium_browser
):
    """``open_panel_set`` side by side, and ``arrange_workspace`` with an exact
    layout of the same three, one call each.

    After each call the terminal's tile keeps its id, its rectangle and its
    panel: a set places service panels and leaves the terminal alone. A
    layout putting two panels in one tile is refused ``one_panel_per_tile``
    and changes nothing.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "c", "d", "e")
        a, c, d, e = (S(x) for x in "acde")
        three = [c, d, e]
        given = {"split": "column", "items": [{"tabs": [c]}, {"tabs": [d]}, {"tabs": [e]}]}

        for layout in (None, given):
            case = "open_panel_set" if layout is None else "arrange_workspace with a layout"
            _restore(page, hub, ONE_SERVICE)
            before = _focus_by_hand(page, a)  # the operator is on A's tile
            _quiet(page, hub)
            posts: list[str] = []

            def answered(request, posts=posts):
                if "/api/panel-op-result" in request.url:
                    posts.append(request.url)

            page.on("request", answered)
            if layout is None:
                result = _tool("open_panel_set", panels=three)
            else:
                result = _tool("arrange_workspace", layout=layout)
            assert result["panels"] == three, (case, result)
            after = _settle(page)
            _assert_returned_is_shown(result, after, case)
            _assert_terminal_untouched(before, after, case)
            assert len(posts) == 1, f"{case}: the dock answered {len(posts)} ops for one call"
            page.remove_listener("request", answered)

            tiles = {_tile_of(after, k) for k in three}
            assert len(tiles) == 3, f"{case}: not one tile each: {after['shape']}"
            if layout is not None:
                assert a not in _keys(after), f"{case}: the layout replaces the service tiles"
                # c over d over e.
                assert _rect(after, c)["y"] < _rect(after, d)["y"] < _rect(after, e)["y"], after[
                    "tiles"
                ]

        # Two panels in one tile: refused by name, and nothing changes.
        start = _restore(page, hub, ONE_SERVICE)
        rev = _tool("list_panels")["rev"]
        crowded = {"split": "column", "items": [{"tabs": [c, d], "showing": d}, {"tabs": [e]}]}
        envelope = _refused("one_panel_per_tile", "arrange_workspace", layout=crowded)
        assert "Nothing changed" in envelope["error_message"], envelope
        assert _settle(page)["shape"] == start["shape"]
        assert hub.app.state.dock_layout_rev == rev
        assert hub.errors == []


def test_a_preset_loads_in_one_call(tmp_path, chromium_browser):
    """``arrange_workspace(preset=...)`` puts the preset's tiles on screen in one call."""
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a")
        a, b, c, d = (S(x) for x in "abcd")
        hub.app.state.panel_presets = [
            {
                "name": "Injection",
                "panels": [b, c, d],
                "layout": {
                    "split": "row",
                    "items": [{"tabs": [b], "size": 2}, {"tabs": [c]}, {"tabs": [d]}],
                },
            }
        ]
        assert [p["name"] for p in _tool("list_panels")["presets"]] == ["Injection"]
        before = _restore(page, hub, ONE_SERVICE)
        posts: list[str] = []
        page.on(
            "request", lambda r: posts.append(r.url) if "/api/panel-op-result" in r.url else None
        )
        result = _tool("arrange_workspace", preset="Injection")
        assert result["preset"] == "Injection"
        after = _settle(page)
        _assert_returned_is_shown(result, after, "preset")
        _assert_terminal_untouched(before, after, "preset")
        assert len(posts) == 1, posts
        assert len({_tile_of(after, k) for k in (b, c, d)}) == 3, after["shape"]
        assert a not in _keys(after), after["shape"]
        for key in (b, c, d):
            expect(_iframe(page, key)).to_be_visible(timeout=5_000)
        # B's tile has twice D's width (weights 2:1:1).
        assert _rect(after, b)["width"] > 1.8 * _rect(after, d)["width"], after["tiles"]
        # Like the operator's Layouts click, the rail now holds exactly the preset's panels.
        for key in (b, c, d):
            expect(rail_entry(page, key)).to_have_count(1, timeout=5_000)
        expect(rail_entry(page, a)).to_have_count(0, timeout=5_000)
        assert hub.errors == []


def test_an_old_shape_panel_presets_fails_the_profile_build(tmp_path, caplog):
    """``osprey build`` on a profile with a list-shaped preset fails and says how to rewrite."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build
    from tests.fixtures.lifecycle_repo import build_exemplar_repo

    repo = build_exemplar_repo(tmp_path / "repo")
    profile = repo / "profile.yml"
    text = profile.read_text()
    assert "panel_presets: {}" in text, "the exemplar no longer spells an empty panel_presets"
    profile.write_text(
        text.replace("panel_presets: {}", "panel_presets:\n  Ops: [okf, channel-finder]")
    )
    caplog.set_level("ERROR")
    result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code != 0, result.output
    # The build reports through its logger, then aborts.
    output = result.output + caplog.text
    assert "panel_presets['Ops'] is the retired list-of-panel-ids form" in output, output
    assert "{split: row, items: [{tabs: [okf]}, {tabs: [channel-finder]}]}" in output, output
    # The rewrite offered is one panel per tile; a tile naming both is never suggested.
    assert "{tabs: [okf, channel-finder]}" not in output, output
    assert not (repo / "build" / "config.yml").exists(), "a refused profile still rendered"


def test_an_agent_open_while_typing_never_covers_the_terminal(tmp_path, chromium_browser):
    """``open_panel`` (with and without ``url``) lands mid-typing; every key reaches the shell.

    The open runs on a worker thread while the operator keeps typing into the
    terminal. Keys typed before, during and after the open arrive in the shell
    in order, the panel never lands in the terminal's tile, and the terminal is
    still shown, uncovered, afterwards.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub, terminal=True)
        _open_services(page, "a", "b")
        a, b, c, d = (S(x) for x in "abcd")
        cases = [
            # A service tile of its own: the panel replaces the one it shows.
            ("own tile", ONE_SERVICE, c),
            # Two service tiles: the panel replaces one of theirs, never the Agent pane.
            ("two tiles", MATRIX_START, d),
        ]
        n = 0
        for case, layout, key in cases:
            for url in (None, f"/panel/{key}/?from=agent"):
                label = f"{case}, {'with' if url else 'without'} url"
                _restore(page, hub, layout)
                _focus_terminal(page)
                first, second, third = f"one{n}", f"two{n}", f"three{n}"
                n += 1
                page.keyboard.type(first)
                call = _Background(_tool, "open_panel", panel_id=key, url=url)
                page.keyboard.type(second)
                result = call.join()
                page.keyboard.type(third)
                page.keyboard.press("Enter")
                _assert_shell_got(hub, first + second + third)
                state = _settle(page)
                _assert_returned_is_shown(result, state, label)
                assert key in _keys(state), f"{label}: {state['shape']}"
                assert _tile_of(state, key) != _tile_of(state, TERMINAL), (
                    f"{label}: the panel landed in the terminal's tile: {state['shape']}"
                )
                _assert_terminal_shown(page, state)
                if case == "own tile":
                    assert state["shape"] == S("row[c,terminal]"), (label, state["shape"])
                    assert _tile_of(state, c) == "t1", (label, state["held"])
        assert hub.errors == []


# ---------------------------------------------------------------------------
# The terminal is never an agent source or close target
# ---------------------------------------------------------------------------


def test_the_terminal_is_refused_as_a_source_or_close_target(tmp_path, chromium_browser):
    """Every agent verb that would move or close the terminal is refused by name.

    The refusal is ``terminal_not_agent_movable``, the dock does not change and
    the rev does not move. The Agent pane is always its own tile: a service
    dropped into its tile (centre or header) is refused ``one_panel_per_tile``,
    as is a drop into any tile, and a person's drop there is admitted nowhere;
    beside it stays allowed.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        b, c = S("b"), S("c")
        start = _restore(page, hub, MATRIX_START)
        rev = _tool("list_panels")["rev"]
        calls = [
            (
                "terminal_not_agent_movable",
                "drop_panel",
                {
                    "source": {"tab": TERMINAL},
                    "target": {"tile": "t1", "edge": "left"},
                    "base_rev": rev,
                },
            ),
            (
                "terminal_not_agent_movable",
                "drop_panel",
                {"source": {"tile": "t3"}, "target": {"outer": "left"}, "base_rev": rev},
            ),
            (
                "terminal_not_agent_movable",
                "drop_panel",
                {
                    "source": {"new": {"panel": TERMINAL}},
                    "target": {"outer": "left"},
                    "base_rev": rev,
                },
            ),
            ("terminal_not_agent_movable", "close_panel", {"panel_id": TERMINAL}),
            ("terminal_not_agent_movable", "open_panel", {"panel_id": TERMINAL}),
            (
                "one_panel_per_tile",
                "drop_panel",
                {
                    "source": {"tab": c},
                    "target": {"tile": "t3", "zone": "centre"},
                    "base_rev": rev,
                },
            ),
            (
                "one_panel_per_tile",
                "drop_panel",
                {
                    "source": {"new": {"panel": S("d")}},
                    "target": {"tile": "t3", "zone": "tab", "index": 0},
                    "base_rev": rev,
                },
            ),
        ]
        for reason, tool, kwargs in calls:
            case = f"{tool}({kwargs})"
            envelope = _refused(reason, tool, **kwargs)
            assert "Nothing changed" in envelope["error_message"], case
            after = _settle(page)
            assert after["shape"] == start["shape"], f"{case}: {after['shape']}"
            assert hub.app.state.dock_layout_rev == rev, f"{case}: the rev moved"
        # A layout naming the terminal is refused under its own name.
        _refused(
            "terminal_in_layout",
            "arrange_workspace",
            layout={"split": "row", "items": [{"tabs": [b]}, {"tabs": [TERMINAL]}]},
        )
        assert _settle(page)["shape"] == start["shape"]

        # A service may still be dropped beside the terminal's tile.
        _restore(page, hub, MATRIX_START)
        rev = _tool("list_panels")["rev"]
        result = _tool(
            "drop_panel", source={"tab": c}, target={"tile": "t3", "edge": "bottom"}, base_rev=rev
        )
        state = _settle(page)
        _assert_returned_is_shown(result, state, "drop beside the terminal's tile")
        assert _tile_of(state, c) != _tile_of(state, TERMINAL), state["shape"]
        assert _rect(state, c)["y"] > _rect(state, TERMINAL)["y"], state["tiles"]
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Revisions: two steps in a row, a stale base, an op mid-drag, an undo
# ---------------------------------------------------------------------------


def _tiles_holding(layout: dict, tabs: list[str]) -> list[str]:
    found: list[str] = []

    def walk(n: dict) -> None:
        if "tile" in n:
            if n["tabs"] == tabs:
                found.append(n["tile"])
        else:
            for item in n["items"]:
                walk(item)

    walk(layout)
    return found


def test_two_steps_in_a_row_and_a_stale_base(tmp_path, chromium_browser):
    """The returned ``rev`` is enough for a second step; a pre-drag ``rev`` is not.

    Two steps in a row read nothing in between: the tile id and ``base_rev``
    of the second call come from the first call's answer. The stale case holds
    the page's report in the network, so the server's rev is still the
    pre-drag one and only the dock knows the layout moved: the dock refuses.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, b, c = (S(x) for x in "abc")

        # Two steps in a row: B to the dock's right edge, then B's new tile to
        # its left edge with the tile id and rev the first drop returned. No
        # second read in between.
        start = _restore(page, hub, MATRIX_START)
        rev = _tool("list_panels")["rev"]
        dropped = _tool("drop_panel", source={"tab": b}, target={"outer": "right"}, base_rev=rev)
        assert dropped["rev"] > rev
        (b_tile,) = _tiles_holding(dropped["layout"], [b])
        moved = _tool(
            "drop_panel", source={"tile": b_tile}, target={"outer": "left"}, base_rev=dropped["rev"]
        )
        assert moved["rev"] > dropped["rev"]
        after = _settle(page)
        _assert_returned_is_shown(moved, after, "drop then drop")
        assert after["shape"] == S("row[b,a,c,terminal]"), after["shape"]
        assert _tile_of(after, b) == b_tile, after["held"]

        # A stale base: a person's drag whose report the server has not had yet.
        # The page's report is held in the network, so the server's rev is still
        # the pre-drag one and only the dock knows the layout moved.
        start = _restore(page, hub, MATRIX_START)
        pre_drag_rev = _tool("list_panels")["rev"]
        held: list = []
        holding = [True]

        def hold(route) -> None:
            if holding[0]:
                held.append(route)
            else:
                route.continue_()

        page.route("**/api/panel-layout", hold)
        try:
            _drag(page, {"grip": _tile_of(start, c)}, _edge_bottom(start, a))
            moved = _settle(page)
            assert moved["shape"] == S("row[column[a,c,b],terminal]"), moved["shape"]
            assert page.evaluate(_HOLDER_JS)["dirty"], "the person's drag was already acked"
            assert hub.app.state.dock_layout_rev == pre_drag_rev
            _refused(
                "stale_base",
                "drop_panel",
                source={"tab": b},
                target={"tile": "t3", "edge": "bottom"},
                base_rev=pre_drag_rev,
            )
            _refused(
                "stale_base",
                "drop_panel",
                source={"tile": "t1"},
                target={"outer": "left"},
                base_rev=pre_drag_rev,
            )
            assert _settle(page)["shape"] == moved["shape"], "a refused op changed the dock"
        finally:
            # Release in order: the held reports go out first, later ones pass
            # straight through, and only then is the handler removed.
            holding[0] = False
            for route in held:
                route.continue_()
            page.unroute("**/api/panel-layout", hold)
        _quiet(page, hub)
        assert hub.app.state.dock_layout_rev > pre_drag_rev
        assert _canon(hub.app.state.dock_layout) == _canon(_settle(page)["layout"])
        assert hub.errors == []


def test_an_agent_op_mid_drag_applies_after_the_drag(tmp_path, chromium_browser):
    """An op that arrives while a person drags waits for the drop, then shows.

    The drag is started from the real grip and left live; ``open_panel`` runs
    meanwhile and answers with D in the layout. Nothing re-lays the dock under
    the person's pointer (no D tile yet, the drag still live); once the person
    drops, both the drop and D are on screen and on the server.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a", "b", "c")
        a, c, d = (S(x) for x in "acd")
        start = _restore(page, hub, MATRIX_START)
        drag = page.evaluate(
            _BEGIN_DRAG_JS,
            {"source": {"grip": _tile_of(start, c)}, "point": _edge_bottom(start, a)},
        )
        assert "error" not in drag, drag
        assert drag["shieldRose"] and drag["dragLive"], drag
        try:
            result = _Background(_tool, "open_panel", panel_id=d).join()
            assert d in _keys(result["layout"]), result
            page.wait_for_timeout(300)
            expect(_bar(page, d)).to_have_count(0)  # nothing re-laid the dock mid-drag
            assert _state(page)["dragLive"], "the drag ended early"
        finally:
            assert page.evaluate(_END_DRAG_JS) == "ok"
        after = _settle(page)
        assert d in _keys(after), f"the op was lost: {after['shape']}"
        expect(_bar(page, d)).to_have_count(1)
        # The person's drop applied too: C's tile (t2, whatever D replaced) sits
        # under A's (t1), above B's (t4).
        column = after["layout"]["items"][0]
        ids = [item.get("tile") for item in column.get("items", [])]
        assert column.get("split") == "column" and ids == ["t1", "t2", "t4"], after["shape"]
        _quiet(page, hub)
        assert _canon(hub.app.state.dock_layout) == _canon(after["layout"])
        assert hub.errors == []


def test_a_person_undoing_an_agent_open_is_reported(tmp_path, chromium_browser):
    """After an agent op, a person's return to the pre-op layout reaches the server.

    The agent opens D, which replaces A in A's tile (recorded through the op
    result); the person clicks A's sidebar row, which replaces D with A and
    puts the dock back exactly where it was before the op. The server, and so
    the agent's next ``list_panels``, must hold that layout and not the op's
    result — otherwise the agent works from a dock that is no longer on
    screen, with a ``rev`` the page still counts as current.
    """
    with _hub(tmp_path) as hub:
        page = _open(chromium_browser, hub)
        _open_services(page, "a")
        a, d = S("a"), S("d")
        before = _restore(page, hub, ONE_SERVICE)
        opened = _tool("open_panel", panel_id=d)
        assert d in _keys(opened["layout"]) and a not in _keys(opened["layout"]), opened
        _settle(page)
        _rail_open(page, a)
        expect(_bar(page, d)).to_have_count(0, timeout=5_000)
        after = _settle(page)
        assert _canon(after["layout"]) == _canon(before["layout"]), after["shape"]
        assert _tile_of(after, a) == _tile_of(before, a), after["held"]
        deadline = time.monotonic() + _SETTLE_S
        listed = _tool("list_panels")
        while d in _keys(listed["layout"] or {}) and time.monotonic() < deadline:
            page.wait_for_timeout(200)
            listed = _tool("list_panels")
        assert d not in _keys(listed["layout"] or {}), (
            f"the person put {a} back, but the agent still reads {d} open: {listed['layout']}"
        )
        assert listed["rev"] > opened["rev"], listed
        assert a in _keys(listed["layout"])
        assert hub.errors == []


# ---------------------------------------------------------------------------
# Which browser tab answers
# ---------------------------------------------------------------------------


def _client_id(page: Page) -> str:
    return page.evaluate(_HOLDER_JS)["client"]


def _wait_for_dock_client(hub: Hub, client: str | None, budget_s: float = 15.0) -> None:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if getattr(hub.app.state, "dock_client_id", None) == client:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"the dock client is {getattr(hub.app.state, 'dock_client_id', None)!r}, not {client!r}"
    )


def test_closing_the_reporting_tab_moves_writes_to_the_other(tmp_path, chromium_browser):
    with _hub(tmp_path) as hub:
        first = _open(chromium_browser, hub)
        _open_services(first, "a")
        second = _open(chromium_browser, hub)
        _open_services(second, "b")  # the second tab reports last: it is current
        one, two = _client_id(first), _client_id(second)
        assert one != two
        _wait_for_dock_client(hub, two)
        a, b, c, d = (S(x) for x in "abcd")
        first_before = _settle(first)

        # A write while both are open goes to the reporting tab only.
        result = _tool("open_panel", panel_id=c)
        assert c in _keys(_settle(second))
        _assert_returned_is_shown(result, _settle(second), "second tab")
        first.wait_for_timeout(300)
        assert _settle(first)["shape"] == first_before["shape"], "a non-current tab changed"

        # Close it: the next write goes to the tab still open.
        second.close()
        _wait_for_dock_client(hub, one)
        result = _tool("open_panel", panel_id=d)
        state = _settle(first)
        _assert_returned_is_shown(result, state, "first tab")
        assert d in _keys(state), state["shape"]

        # None open: refused no_dock_client, nothing recorded.
        rev = hub.app.state.dock_layout_rev
        first.close()
        _refused("no_dock_client", "open_panel", panel_id=b)
        assert hub.app.state.dock_layout_rev == rev
        assert hub.errors == []


# ---------------------------------------------------------------------------
# ARIEL and Phoebus open through the op path
# ---------------------------------------------------------------------------


def test_ariel_and_phoebus_open_through_the_op_path_in_one_tab(
    tmp_path, chromium_browser, monkeypatch
):
    """ARIEL's and Phoebus's own open paths each land in one dock tile, in one browser tab.

    ARIEL opens through its tool's ``open {member: ariel, url}`` helper, Phoebus
    through ``phoebus_open_panel`` (its bridge stubbed; ``OSPREY_SERVER_NAME``
    names its panel). Opening twice keeps one tile of each; Phoebus's open
    replaces ARIEL in the service tile, whose iframe stays cached and hidden;
    the dock client answers each op once through ``/api/panel-op-result``; the
    other browser tab, which is not the dock client, never changes.
    """
    from osprey.mcp_server.ariel.tools.entry import _focus_ariel_panel
    from osprey.mcp_server.phoebus.tools import bridge_tools

    monkeypatch.setenv("OSPREY_SERVER_NAME", PHOEBUS)
    with _hub(tmp_path) as hub:
        other = _open(chromium_browser, hub)
        other_before = _settle(other)
        page = _open(chromium_browser, hub)
        _open_services(page, "a")
        _restore(page, hub, ONE_SERVICE)
        _wait_for_dock_client(hub, _client_id(page))
        answers: list[str] = []
        page.on(
            "request",
            lambda r: (
                answers.append(r.post_data or "") if "/api/panel-op-result" in r.url else None
            ),
        )
        other_posts: list[str] = []
        other.on(
            "request",
            lambda r: other_posts.append(r.url) if "/api/panel-op" in r.url else None,
        )

        for entry in (7, 8):
            _run_async(_focus_ariel_panel, f"/panel/ariel/#entry?id={entry}")
            state = _settle(page)
            assert _keys(state).count(ARIEL) == 1, state["shape"]
            expect(_bar(page, ARIEL)).to_have_count(1)
            expect(_iframe(page, ARIEL)).to_have_count(1)
            _assert_terminal_shown(page, state)
        assert len(answers) == 2 and all('"layout"' in body for body in answers), answers

        with (
            patch.object(bridge_tools, "_resolve_panel_resource", return_value="/x/overview.bob"),
            patch.object(
                bridge_tools,
                "_http_post_open",
                return_value=(200, {"id": "d-1", "resource": "/x/overview.bob", "ready": True}),
            ),
        ):
            for _ in range(2):
                raw = _run_async(get_tool_fn(bridge_tools.phoebus_open_panel), name="overview")
                assert json.loads(raw)["focused"] is True, raw
                state = _settle(page)
                assert _keys(state).count(PHOEBUS) == 1, state["shape"]
                expect(_bar(page, PHOEBUS)).to_have_count(1)
        assert len(answers) == 4, answers
        assert PHOEBUS in _keys(state) and ARIEL not in _keys(state), state["shape"]
        expect(_iframe(page, ARIEL)).to_have_count(1)
        expect(_iframe(page, ARIEL)).to_be_hidden()

        assert other_posts == [], f"the other browser tab answered ops: {other_posts}"
        assert _settle(other)["shape"] == other_before["shape"]
        expect(_bar(other, ARIEL)).to_have_count(0)
        assert hub.errors == []
