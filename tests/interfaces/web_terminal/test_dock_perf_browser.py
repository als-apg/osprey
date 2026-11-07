"""A drag sweep over a busy dock adds no long task.

This mirrors the CANOPY viewer's dock performance check in the OSPREY web
terminal, with live services.

The fixture. Four custom panels, ``perf-plot-1..4``, each backed by its own live
HTTP service on loopback. A service serves one page: a full-size SVG plot of
four curves that polls the service's ``data`` endpoint every 100 ms and
redraws, while the service random-walks the curves on every request. So all
four plots repaint, over the network, while the drag runs. The panels are
opened from the sidebar as a person would (each row click replaces the panel
of the service tile used last, so each gets its iframe), then arranged into a
fixed 2x2 grid (a row of two columns, two tiles each) beside the agent pane's
tile through the layout holder — the same entry a saved dock takes on reload.

The measurement. A ``longtask`` PerformanceObserver is installed before the hub
loads. Two 5 s windows are compared, timed inside the page so no Playwright
round-trip lands in them: 5 s of no drag, then a scripted 5 s sweep. The sweep
is a real HTML5 drag of the first tile's header (Playwright's mouse never
raises HTML5 drag-and-drop, so the gesture is dispatched in the page with a
real DataTransfer): ``dragstart`` on the header, then, once the drag shield is
up, a ``dragover`` every frame at the next point of a fixed tour of every zone
of every plot tile (centre, the four edge bands, and the header bar, which is
no drop target but is still hit-tested), the borders between tiles and the
dock's outer band, with a ``dragenter`` whenever the hit element changes; it
ends with ``dragleave``, Escape and
``dragend``, so no drop lands and the layout stays as arranged. A long task
counts for a window when it starts inside it.

Always asserted: the dock holds the 2x2 grid before and after, every plot is
live, the sweep really was a drag (the shield rose and a drop preview was
painted), and the sweep visited every zone. The counts are written to
``dock-perf.json`` under ``DOCK_PERF_OUT`` when that is set, else under the
test's own temporary directory — never anywhere the next run reads. Timing is
only meaningful on an otherwise idle machine, so the long-task assertion (the
drag window has no more long tasks over 50 ms than the idle window) runs only
under ``PERF_GATE=1``.

Run:

    PERF_GATE=1 uv run pytest tests/interfaces/web_terminal/test_dock_perf_browser.py -rs

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
import os
import random
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.interfaces._browser import open_hub_page, rail_entry, wait_for_dock_settled
from tests.interfaces.web_terminal.test_panels_browser import _live_server

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover - the chromium fixture skips first
    Page = object  # type: ignore[assignment,misc]

pytestmark = [pytest.mark.browser, pytest.mark.slow]

WINDOW_MS = 5000
#: A long task is one over 50 ms; the gate allows the drag none beyond the idle window's.
LONG_TASK_MS = 50
PERF_GATE = os.environ.get("PERF_GATE") == "1"
#: Large enough that four plot tiles and the agent pane clear the 40 px floor with room.
VIEWPORT = {"width": 1600, "height": 1000}

IDS = ["perf-plot-1", "perf-plot-2", "perf-plot-3", "perf-plot-4"]
#: The fixed 2x2 grid beside the agent pane, in the holder's dock form.
GRID = {
    "split": "row",
    "items": [
        {
            "split": "column",
            "items": [
                {"tile": "p1", "tabs": [IDS[0]], "showing": IDS[0]},
                {"tile": "p2", "tabs": [IDS[1]], "showing": IDS[1]},
            ],
            "size": 0.35,
        },
        {
            "split": "column",
            "items": [
                {"tile": "p3", "tabs": [IDS[2]], "showing": IDS[2]},
                {"tile": "p4", "tabs": [IDS[3]], "showing": IDS[3]},
            ],
            "size": 0.35,
        },
        {"tile": "tt", "tabs": ["terminal"], "showing": "terminal", "size": 0.3},
    ],
}
SHAPE = f"row[column[{IDS[0]},{IDS[1]}],column[{IDS[2]},{IDS[3]}],terminal]"

# ---------------------------------------------------------------------------
# The live services
# ---------------------------------------------------------------------------

_POINTS = 200
_CURVES = 4

#: One live plot: an SVG of four polylines, redrawn from the service's ``data``
#: endpoint every 100 ms. The proxy serves a panel as an opaque origin, so the page
#: reads its service directly (CORS) and reports each redraw to the hub by
#: ``postMessage``; that is how a test sees it live.
_PLOT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>perf plot</title>
<style>html,body{margin:0;height:100%;background:#fff}svg{display:block;width:100%;height:100%}</style>
</head><body>
<svg id="plot" viewBox="0 0 600 360" preserveAspectRatio="none">
  <polyline fill="none" stroke="#1f77b4" stroke-width="2"/>
  <polyline fill="none" stroke="#ff7f0e" stroke-width="2"/>
  <polyline fill="none" stroke="#2ca02c" stroke-width="2"/>
  <polyline fill="none" stroke="#d62728" stroke-width="2"/>
</svg>
<script>
  const lines = [...document.querySelectorAll('polyline')];
  const url = '__DATA_URL__';
  let frames = 0;
  async function tick() {
    try {
      const curves = await (await fetch(url, { cache: 'no-store' })).json();
      curves.forEach((ys, c) => {
        const step = 600 / (ys.length - 1);
        lines[c].setAttribute('points', ys.map((y, i) => `${(i * step).toFixed(1)},${(180 - y).toFixed(1)}`).join(' '));
      });
      frames += 1;
      parent.postMessage({ type: 'osprey-perf-plot-frame', frames }, '*');
    } catch (e) { /* a missed poll: the next one redraws */ }
    setTimeout(tick, 100);
  }
  tick();
</script>
</body></html>
"""


@contextmanager
def _plot_service() -> Iterator[str]:
    """A live data service: four random-walk curves, advanced on every read.

    Yields:
        The service's base URL.
    """
    lock = threading.Lock()
    curves = [[0.0] * _POINTS for _ in range(_CURVES)]
    rng = random.Random()

    def advance() -> list[list[float]]:
        with lock:
            for ys in curves:
                last = max(-160.0, min(160.0, ys[-1] + rng.uniform(-12.0, 12.0)))
                ys.pop(0)
                ys.append(last)
            return [list(ys) for ys in curves]

    base = ""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path.endswith("/data"):
                body, kind = json.dumps(advance()).encode(), "application/json"
            elif path.endswith("/health"):
                body, kind = b"ok", "text/plain"
            else:
                page = _PLOT_PAGE.replace("__DATA_URL__", f"{base}/data")
                body, kind = page.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-request stderr logging
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    base = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# In-page scripts
# ---------------------------------------------------------------------------

#: Installed before the hub's own scripts: every long task, with its start and duration.
_LONG_TASK_OBSERVER = """
(() => {
  if (window !== window.top) return;
  window.__perfLongTasks = [];
  // Each plot reports its redraws; counted by the sending window.
  window.__perfFrames = new Map();
  window.addEventListener('message', (e) => {
    if (e.data?.type === 'osprey-perf-plot-frame' && e.source) {
      window.__perfFrames.set(e.source, e.data.frames);
    }
  });
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        window.__perfLongTasks.push({ start: e.startTime, ms: e.duration });
      }
    }).observe({ type: 'longtask', buffered: true });
  } catch {
    // No longtask support: the test fails on the missing observer below.
  }
})();
"""

#: Put the grid in the holder and show every tile's plot, as a saved dock would on reload.
_ARRANGE_JS = """
async (layout) => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const { focusPanel } = await import('/static/js/dock-iframe.js');
  const holder = getLayoutHolder();
  if (!holder) return 'no layout holder';
  if (!holder.restore(layout)) return 'refused';
  const walk = (n) => {
    if ('tile' in n) { if (n.showing !== 'terminal') focusPanel(n.showing); }
    else n.items.forEach(walk);
  };
  walk(layout);
  return 'ok';
}
"""

#: The holder's layout as a split shape of showing keys, e.g. ``row[column[a,b],terminal]``.
_SHAPE_JS = """
async () => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const layout = getLayoutHolder()?.layout ?? null;
  const shape = (n) => 'tile' in n
    ? (n.tabs.length === 1 ? n.showing : `[${n.tabs.join('|')}]`)
    : `${n.split}[${n.items.map(shape).join(',')}]`;
  return layout ? shape(layout) : 'null';
}
"""

#: The holder's layout and drag state, for the before/after comparison.
_HOLDER_JS = """
async () => {
  const { getLayoutHolder } = await import('/static/js/dock-apply.js');
  const holder = getLayoutHolder();
  return { layout: JSON.stringify(holder?.layout ?? null), dragLive: !!holder?.dragLive };
}
"""

#: How many times each plot has redrawn, by panel id (-1: no readable document).
_FRAMES_JS = """
(ids) => Object.fromEntries(ids.map((id) => {
  const iframe = document.querySelector(`.dock-iframe-overlay iframe[data-panel-id="${id}"]`);
  const frames = iframe?.contentWindow ? window.__perfFrames?.get(iframe.contentWindow) : undefined;
  return [id, frames ?? -1];
}))
"""

#: Long tasks over the floor that started inside [start, end).
_LONG_TASKS_IN_JS = """
({ start, end, min }) => (window.__perfLongTasks ?? [])
  .filter((t) => t.start >= start && t.start < end && t.ms > min)
  .map((t) => Math.round(t.ms * 10) / 10)
"""

#: WINDOW_MS of no drag, timed in the page.
_IDLE_JS = """
async (ms) => {
  const startMs = performance.now();
  await new Promise((r) => setTimeout(r, ms));
  return { startMs, endMs: performance.now() };
}
"""

#: WINDOW_MS of a scripted drag of the first plot's header over every zone, never dropped.
_SWEEP_JS = r"""
async ({ ms, ids }) => {
  const sleep = (t) => new Promise((r) => setTimeout(r, t));
  const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
  const root = document.querySelector('#dock-root');
  if (!root) throw new Error('sweep: no dock root');
  const dock = root.getBoundingClientRect();
  const barOf = (key) => root.querySelector(`.tile-tab[data-tile][data-panel-key="${CSS.escape(key)}"]`);
  const groupOf = (key) => barOf(key)?.closest('.dv-groupview') ?? null;

  // The tour: every zone of every plot tile, the borders between tiles, the outer band.
  const tour = [];
  const rects = {};
  for (const key of ids) {
    const bar = barOf(key);
    const group = groupOf(key);
    if (!bar || !group) throw new Error(`sweep: no tile for '${key}'`);
    rects[key] = group.getBoundingClientRect();
    const tile = bar.dataset.tile;
    const b = bar.getBoundingClientRect();
    tour.push({ x: b.x + 24, y: b.y + b.height / 2, zone: `${key}:bar` });
    tour.push({ x: b.x + b.width - 6, y: b.y + b.height / 2, zone: `${key}:bar` });
    const content = group.querySelector('.dv-content-container');
    if (!content) throw new Error(`sweep: no content for tile ${tile}`);
    const c = content.getBoundingClientRect();
    const cx = c.x + c.width / 2;
    const cy = c.y + c.height / 2;
    const inset = Math.min(20, c.width / 8, c.height / 8);
    tour.push({ x: cx, y: cy, zone: `${key}:center` });
    tour.push({ x: c.x + inset, y: cy, zone: `${key}:left` });
    tour.push({ x: c.x + c.width - inset, y: cy, zone: `${key}:right` });
    tour.push({ x: cx, y: c.y + inset, zone: `${key}:top` });
    tour.push({ x: cx, y: c.y + c.height - inset, zone: `${key}:bottom` });
  }
  // Borders: the grid's sashes and the one between the grid and the agent pane,
  // at several points along each.
  const [p1, p2, p3, p4] = ids.map((k) => rects[k]);
  const terminal = document.querySelector('#dock-root .tile-tab-terminal')
    ?.closest('.dv-groupview')?.getBoundingClientRect();
  for (const f of [0.2, 0.4, 0.6, 0.8]) {
    const vx = (p1.right + p3.left) / 2;
    tour.push({ x: vx, y: p1.top + (p2.bottom - p1.top) * f, zone: 'border:vertical' });
    tour.push({ x: p1.left + p1.width * f, y: (p1.bottom + p2.top) / 2, zone: 'border:left' });
    tour.push({ x: p3.left + p3.width * f, y: (p3.bottom + p4.top) / 2, zone: 'border:right' });
    if (terminal) {
      const tx = (p3.right + terminal.left) / 2;
      tour.push({ x: tx, y: p3.top + (p4.bottom - p3.top) * f, zone: 'border:terminal' });
    }
  }
  // The outer band: just inside each dock edge.
  for (const f of [0.25, 0.5, 0.75]) {
    tour.push({ x: dock.x + 3, y: dock.y + dock.height * f, zone: 'outer:left' });
    tour.push({ x: dock.x + dock.width - 3, y: dock.y + dock.height * f, zone: 'outer:right' });
    tour.push({ x: dock.x + dock.width * f, y: dock.y + 3, zone: 'outer:top' });
    tour.push({ x: dock.x + dock.width * f, y: dock.y + dock.height - 3, zone: 'outer:bottom' });
  }

  // The first plot's header is the drag handle: dockview's own tab drag, the tile.
  const source = barOf(ids[0])?.closest('.dv-tab');
  if (!source) throw new Error(`sweep: no header for '${ids[0]}'`);
  const dataTransfer = new DataTransfer();
  const init = (x, y) => ({
    bubbles: true, cancelable: true, composed: true, dataTransfer, clientX: x, clientY: y,
  });
  const kinds = new Set();
  const zones = new Set();
  const inDock = new Set();
  const watchPreview = () => {
    for (const el of document.querySelectorAll('.dock-drop-preview__rect')) {
      for (const cls of el.classList) {
        if (cls.startsWith('dock-drop-preview__rect--')) kinds.add(cls.slice(25));
      }
    }
  };
  const shieldUp = () => !!document.querySelector('.main-container.dock-dragging');

  const startMs = performance.now();
  const s = source.getBoundingClientRect();
  source.dispatchEvent(new DragEvent('dragstart', init(s.x + s.width / 2, s.y + s.height / 2)));
  // Hit-test only once the shield is up, so the plots' iframes let the points through.
  let shieldRose = false;
  const shieldDeadline = performance.now() + 1000;
  while (performance.now() < shieldDeadline) {
    if (shieldUp()) { shieldRose = true; break; }
    await sleep(16);
  }
  const iframesShielded = [...document.querySelectorAll('.dock-iframe-overlay iframe')]
    .every((f) => f.style.pointerEvents === 'none');
  let last = null;
  let lastPoint = { x: s.x, y: s.y };
  let dragovers = 0;
  let i = 0;
  while (performance.now() - startMs < ms) {
    const stop = tour[i % tour.length];
    i += 1;
    const target = document.elementFromPoint(stop.x, stop.y);
    if (target) {
      if (target !== last) {
        last?.dispatchEvent(new DragEvent('dragleave', init(stop.x, stop.y)));
        target.dispatchEvent(new DragEvent('dragenter', init(stop.x, stop.y)));
        last = target;
      }
      target.dispatchEvent(new DragEvent('dragover', init(stop.x, stop.y)));
      dragovers += 1;
      zones.add(stop.zone);
      if (root.contains(target)) inDock.add(stop.zone);
      lastPoint = stop;
    }
    await frame();
    watchPreview();
  }
  // Cancel: no drop, so the layout stays as arranged.
  last?.dispatchEvent(new DragEvent('dragleave', init(lastPoint.x, lastPoint.y)));
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  (source.isConnected ? source : document).dispatchEvent(
    new DragEvent('dragend', init(lastPoint.x, lastPoint.y)));
  const endMs = performance.now();
  return {
    startMs,
    endMs,
    dragovers,
    points: tour.length,
    zones: [...zones].sort(),
    zonesInDock: [...inDock].sort(),
    previewKinds: [...kinds].sort(),
    shieldRose,
    iframesShielded,
  };
}
"""

_TILE_ZONES = ("bar", "center", "left", "right", "top", "bottom")
_BORDER_ZONES = (
    "border:vertical",
    "border:left",
    "border:right",
    "border:terminal",
    "outer:left",
    "outer:right",
    "outer:top",
    "outer:bottom",
)


def _long_tasks_in(page: Page, start: float, end: float) -> list[float]:
    return page.evaluate(_LONG_TASKS_IN_JS, {"start": start, "end": end, "min": LONG_TASK_MS})


def _idle_window(page: Page) -> dict:
    window = page.evaluate(_IDLE_JS, WINDOW_MS)
    # Let the observer deliver tasks that ended at the window's edge.
    page.wait_for_timeout(200)
    window["longTasks"] = _long_tasks_in(page, window["startMs"], window["endMs"])
    return window


def _sweep_window(page: Page) -> dict:
    sweep = page.evaluate(_SWEEP_JS, {"ms": WINDOW_MS, "ids": IDS})
    page.wait_for_timeout(200)
    sweep["longTasks"] = _long_tasks_in(page, sweep["startMs"], sweep["endMs"])
    return sweep


def _out_dir(tmp_path: Path) -> Path:
    override = os.environ.get("DOCK_PERF_OUT")
    out = Path(override).resolve() if override else tmp_path / "dock-perf"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_drag_sweep_over_a_2x2_dock_of_live_plots_adds_no_long_task(tmp_path, chromium_browser):
    """A 5 s drag sweep over a 2x2 dock of live plots adds no long task over 50 ms."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with ExitStack() as stack:
        services = [stack.enter_context(_plot_service()) for _ in IDS]
        custom = [
            {
                "id": pid,
                "label": f"PERF PLOT {n}",
                "url": url,
                "healthEndpoint": None,
                "path": "/",
            }
            for n, (pid, url) in enumerate(zip(IDS, services, strict=True), start=1)
        ]
        base_url, app = stack.enter_context(
            _live_server(workspace, enabled_panels={"artifacts"}, custom_panels=custom)
        )
        app.state.visible_panels = ["artifacts", *IDS]

        context = chromium_browser.new_context(viewport=VIEWPORT)
        stack.callback(context.close)
        context.add_init_script(_LONG_TASK_OBSERVER)
        page = open_hub_page(context, base_url)
        wait_for_dock_settled(page)

        # Open every plot from the sidebar, as a person would; each gets its iframe.
        for pid in IDS:
            rail_entry(page, pid, enabled_only=True).click()
            expect(
                page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{pid}"]')
            ).to_be_visible(timeout=10_000)

        # Arrange them into the fixed 2x2 grid beside the agent pane.
        assert page.evaluate(_ARRANGE_JS, GRID) == "ok"
        page.wait_for_function(
            "(ids) => ids.every((id) => document.querySelector("
            '`.tile-tab[data-tile][data-panel-key="${id}"]`))',
            arg=IDS,
            timeout=5_000,
        )
        assert page.evaluate(_SHAPE_JS) == SHAPE
        assert page.locator(".tile-tab[data-tile]").count() == 5  # four plots + the agent pane
        for pid in IDS:
            expect(
                page.locator(f'.dock-iframe-overlay iframe[data-panel-id="{pid}"]')
            ).to_be_visible(timeout=5_000)

        # Every plot is live: each redraws from its service while we watch.
        before = page.evaluate(_FRAMES_JS, IDS)
        page.wait_for_function(
            f"([ids, before]) => {{ const now = ({_FRAMES_JS})(ids);"
            " return ids.every((id) => now[id] > before[id] + 5); }",
            arg=[IDS, before],
            timeout=10_000,
        )

        observing = page.evaluate(
            "() => PerformanceObserver.supportedEntryTypes.includes('longtask')"
            " && Array.isArray(window.__perfLongTasks)"
        )
        assert observing, "longtask PerformanceObserver installed"
        held = page.evaluate(_HOLDER_JS)
        # Let the first paints and data settle before either window opens.
        page.wait_for_timeout(1000)

        idle = _idle_window(page)
        frames_before_sweep = page.evaluate(_FRAMES_JS, IDS)
        sweep = _sweep_window(page)
        frames_after_sweep = page.evaluate(_FRAMES_JS, IDS)

        # The sweep was a real drag that the dock answered, over every zone,
        # and it left the layout alone.
        assert sweep["shieldRose"], "the drag shield rose"
        assert sweep["iframesShielded"], "the drag shield reached every plot's iframe"
        assert sweep["previewKinds"], "a drop preview was painted"
        for pid in IDS:
            for zone in _TILE_ZONES:
                assert f"{pid}:{zone}" in sweep["zonesInDock"], f"the sweep visited {pid}:{zone}"
        for zone in _BORDER_ZONES:
            assert zone in sweep["zonesInDock"], f"the sweep visited {zone}"
        # The plots kept repainting through the drag.
        for pid in IDS:
            assert frames_after_sweep[pid] > frames_before_sweep[pid] + 10, (
                f"{pid} stopped updating during the drag"
            )
        page.wait_for_function(
            "() => !document.querySelector('.main-container.dock-dragging')", timeout=2_000
        )
        page.wait_for_function(
            "async () => !(await import('/static/js/dock-apply.js')).getLayoutHolder()?.dragLive",
            timeout=2_000,
        )
        after = page.evaluate(_HOLDER_JS)
        assert after["layout"] == held["layout"], "the cancelled drag changed the layout"
        assert page.evaluate(_SHAPE_JS) == SHAPE

        report = {
            "windowMs": WINDOW_MS,
            "longTaskMs": LONG_TASK_MS,
            "idle": {
                "measuredMs": round(idle["endMs"] - idle["startMs"]),
                "longTasks": len(idle["longTasks"]),
                "durationsMs": idle["longTasks"],
            },
            "drag": {
                "measuredMs": round(sweep["endMs"] - sweep["startMs"]),
                "longTasks": len(sweep["longTasks"]),
                "durationsMs": sweep["longTasks"],
                "dragovers": sweep["dragovers"],
                "tourPoints": sweep["points"],
                "zones": len(sweep["zones"]),
                "previewKinds": sweep["previewKinds"],
            },
            "addedLongTasks": max(0, len(sweep["longTasks"]) - len(idle["longTasks"])),
            "gated": PERF_GATE,
            "browser": chromium_browser.version,
            "at": datetime.now(UTC).isoformat(),
        }
        out = _out_dir(tmp_path) / "dock-perf.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\ndock perf: {json.dumps(report)}\nwritten to {out}")

        if not PERF_GATE:
            return
        assert len(sweep["longTasks"]) <= len(idle["longTasks"]), (
            f"long tasks > {LONG_TASK_MS} ms during the drag ({sweep['longTasks']}) "
            f"vs idle ({idle['longTasks']})"
        )
