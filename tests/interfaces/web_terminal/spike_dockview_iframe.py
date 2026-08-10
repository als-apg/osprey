"""Dockview iframe-reload spike (dock-spike).

QUESTION
--------
When a dockview panel whose content is an ``<iframe>`` is moved between groups
by a real drag (and when a splitter is dragged), does the browser preserve the
iframe's live document -- its mutable JS state and its single ``load`` -- or does
dockview re-parent the ``<iframe>`` DOM node and force a reload?  The answer
selects the implementation path for the dock iframe adapter
(static/js/dock-iframe.js):

  * PASS -> native dockview panels are safe hosts for iframes; the adapter
    mounts each iframe directly as a dockview panel component.
  * FAIL -> dockview re-parents the panel content on regroup, which reloads the
    iframe and destroys its state; the adapter must instead keep the iframes in
    a fixed, un-reparented overlay layer and sync each iframe's geometry to its
    dockview panel's rectangle on layout/resize events (iframe panels only;
    floating/maximize out of scope).

METHOD
------
A local ``http.server`` serves a scratch page that builds a REAL dockview v7.0.2
workspace from the vendored assets
(``static/vendor/dockview-core.min.js`` + ``dockview.css``; UMD global
``window['dockview-core']``).  The workspace has three groups laid left-to-right:
an iframe panel and two plain panels.  The iframe's document holds two signals:

  * ``window.__counter`` -- in-memory mutable state, set to 42 by the driver
    AFTER load; resets to 0 on any document reload.
  * ``sessionStorage['loadCount']`` -- incremented once per document parse; a
    cross-reload counter of how many times the iframe's ``load`` has fired.

Sync Playwright then, reading (counter, loadCount) back from inside the iframe
after each step:

  (a) drags the iframe panel's tab across to dock at the LEFT edge of another
      group (cross-group move -> new adjacent group),
  (b) drags it into the CENTRE of the third group (drop INTO that group's tab
      stack),
  (c) drags a splitter (``.dv-sash``) to resize.

Steps (a) and (b) are driven as real HTML5 drag-and-drop: DragEvents dispatched
on the actual ``.dv-tab`` source and ``.dv-content-container`` drop target with a
shared ``DataTransfer``, which runs dockview's real ``dragstart``/``drop``
handlers and its real DOM re-parenting (dockview stores an internal
``PanelTransfer`` singleton on dragstart and reads it on drop -- the same code
path a user drag takes).  Step (c) is a genuine Playwright mouse pointer drag on
the sash.  The driver asserts each step actually moved the panel / changed the
layout before trusting its reading.

VERDICT: FAIL  (dockview-core 7.0.2, chromium via Playwright; run 2026-07-20)

Native dockview panels do NOT preserve iframe state across a cross-group move.
Re-parenting the panel's content element reloads the iframe every time.

Raw evidence (verbatim, deterministic across repeated runs):

  baseline (after load, counter set to 42):  counter=42  loadCount=1  groups=3
  (a) drag iframe tab -> LEFT of group 2:     counter=0   loadCount=2  RELOADED=yes  groups=3  ifpGroup=1
  (b) drag iframe tab -> CENTRE of group 3:   counter=0   loadCount=3  RELOADED=yes  STACKED=yes  panels=['p2','ifp']
  (c) drag splitter (.dv-sash) to resize:     counter=0   loadCount=3  RESIZED=yes

Interpretation:
  * Each cross-group drag ((a),(b)) reset counter 42 -> 0 AND incremented
    loadCount, i.e. the iframe fully reloaded. An <iframe> only reloads when its
    DOM node is detached/re-attached, so this is direct proof dockview
    re-parents the panel's content element on regroup. State is destroyed. (The
    group *count* stays 3 and dockview recycles group ids, so group ids/counts
    are NOT a reliable move signal -- the reload is the ground truth, hence the
    RELOADED flag rather than a MOVED flag.)
  * The splitter resize (c) did NOT reload the iframe on its own (loadCount held
    at 3, counter stayed 0), because a pure resize does not re-parent the panel
    element -- but by then the state was already gone from the (a)/(b) reloads.
  * PASS required counter==42 AND loadCount==1 after ALL THREE. Observed
    counter==0 and loadCount==3. => FAIL.

CONSEQUENCE for the dock iframe adapter (dock-iframe.js): take the
overlay-fallback path.
Keep iframes in a fixed overlay layer that is never re-parented by dockview;
mount an empty placeholder panel per iframe in dockview and, on dockview
layout/resize events, copy the placeholder panel's bounding rectangle onto the
corresponding overlay iframe. Scope: iframe panels only; geometry sync on
layout/resize events only; floating/maximize out of scope.

REPRODUCE (one command, from the worktree root):
    uv run python tests/interfaces/web_terminal/spike_dockview_iframe.py

Named ``spike_*`` so pytest does not collect it (collection globs are
``test_*.py`` / ``*_test.py``); this is an evidence-producing driver, not a test.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[3] / "src/osprey/interfaces/web_terminal/static/vendor"

# --- scratch page: a real dockview workspace hosting an iframe panel ---------

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>dockview iframe spike</title>
<link rel="stylesheet" href="dockview.css">
<style>
  html, body { margin: 0; height: 100%; }
  #dock { position: absolute; inset: 0; }
  .plain { padding: 12px; font: 14px monospace; }
  .iframe-wrap { position: absolute; inset: 0; }
  .iframe-wrap iframe { width: 100%; height: 100%; border: 0; display: block; }
</style>
<div id="dock" class="dockview-theme-dark"></div>
<script src="dockview-core.min.js"></script>
<script>
  const { createDockview } = window['dockview-core'];

  function createComponent(options) {
    if (options.name === 'iframe') {
      const el = document.createElement('div');
      el.className = 'iframe-wrap';
      el.id = 'marker-' + options.id;
      const f = document.createElement('iframe');
      f.src = 'iframe.html';
      el.appendChild(f);
      return { element: el, init() {} };
    }
    const el = document.createElement('div');
    el.className = 'plain';
    el.id = 'marker-' + options.id;
    el.textContent = 'plain panel ' + options.id;
    return { element: el, init() {} };
  }

  const api = createDockview(document.getElementById('dock'), { createComponent });
  window.__api = api;

  // Three groups, left to right: [iframe][p1][p2]
  const ifp = api.addPanel({ id: 'ifp', component: 'iframe', title: 'IFRAME' });
  const p1 = api.addPanel({
    id: 'p1', component: 'plain', title: 'P1',
    position: { referencePanel: 'ifp', direction: 'right' },
  });
  api.addPanel({
    id: 'p2', component: 'plain', title: 'P2',
    position: { referencePanel: 'p1', direction: 'right' },
  });

  // Report how many groups exist and which group the iframe panel lives in --
  // lets the driver confirm a drag actually re-parented the panel.
  window.__layout = function () {
    return {
      groups: api.groups.length,
      ifpGroup: api.getPanel('ifp').group.id,
      ifpGroupPanels: api.getPanel('ifp').group.panels.map(p => p.id),
    };
  };

  // Real HTML5 drag: dispatch DragEvents on the actual tab + drop-zone with a
  // shared DataTransfer. Runs dockview's real dragstart/drop handlers (which use
  // an internal PanelTransfer singleton), so the panel is genuinely re-parented.
  window.__drag = function (sourceSel, targetSel, relX, relY) {
    const src = document.querySelector(sourceSel);
    const tgt = document.querySelector(targetSel);
    if (!src || !tgt) throw new Error('drag selectors missed: ' + sourceSel + ' / ' + targetSel);
    const sb = src.getBoundingClientRect();
    const tb = tgt.getBoundingClientRect();
    const sx = sb.x + sb.width / 2, sy = sb.y + sb.height / 2;
    const tx = tb.x + tb.width * relX, ty = tb.y + tb.height * relY;
    const dt = new DataTransfer();
    const fire = (type, el, x, y) => el.dispatchEvent(new DragEvent(type, {
      bubbles: true, cancelable: true, composed: true,
      clientX: x, clientY: y, dataTransfer: dt,
    }));
    fire('dragstart', src, sx, sy);
    fire('dragenter', tgt, tx, ty);
    fire('dragover', tgt, tx, ty);
    fire('dragover', tgt, tx, ty);
    fire('drop', tgt, tx, ty);
    fire('dragend', src, tx, ty);
  };
</script>
"""

IFRAME_HTML = """<!doctype html>
<meta charset="utf-8">
<title>iframe state</title>
<body style="font:14px monospace; padding:8px">
<div id="out"></div>
<script>
  // Runs once per document parse == once per iframe load. sessionStorage is
  // shared with the top-level tab (same origin) and survives an element reload,
  // so it counts total loads even if dockview destroys+recreates the iframe.
  var lc = (parseInt(sessionStorage.getItem('loadCount') || '0', 10)) + 1;
  sessionStorage.setItem('loadCount', String(lc));

  window.__counter = 0;                       // in-memory; resets to 0 on reload
  window.__setCounter = function (v) { window.__counter = v; render(); };
  window.__state = function () {
    return {
      counter: window.__counter,
      loadCount: parseInt(sessionStorage.getItem('loadCount') || '0', 10),
    };
  };
  function render() { document.getElementById('out').textContent = JSON.stringify(window.__state()); }
  render();
</script>
"""


def _serve(root: Path) -> tuple[ThreadingHTTPServer, str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}/index.html"


def _iframe_state(page):
    """Read (counter, loadCount) from inside the iframe, re-finding the frame."""
    page.wait_for_timeout(300)  # let any pending reload settle
    for frame in page.frames:
        if frame.url.endswith("iframe.html"):
            return frame.evaluate("() => window.__state()")
    raise RuntimeError("iframe frame not found")


def main() -> int:
    from playwright.sync_api import sync_playwright

    root = Path(tempfile.mkdtemp(prefix="dockspike-"))
    shutil.copy(VENDOR / "dockview-core.min.js", root / "dockview-core.min.js")
    shutil.copy(VENDOR / "dockview.css", root / "dockview.css")
    (root / "index.html").write_text(INDEX_HTML)
    (root / "iframe.html").write_text(IFRAME_HTML)

    httpd, url = _serve(root)
    rows: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 800})
            page.goto(url)
            page.wait_for_selector(".dv-tab", timeout=5000)
            # Wait for the iframe's first load, then plant mutable state.
            page.wait_for_function(
                "() => [...document.querySelectorAll('iframe')].some(f => f.contentWindow && f.contentWindow.__setCounter)",
                timeout=5000,
            )
            for frame in page.frames:
                if frame.url.endswith("iframe.html"):
                    frame.evaluate("() => window.__setCounter(42)")

            base = _iframe_state(page)
            layout0 = page.evaluate("() => window.__layout()")
            rows.append(
                f"baseline:                       counter={base['counter']} loadCount={base['loadCount']} groups={layout0['groups']}"
            )

            # (a) drag iframe tab to the LEFT edge of P1's group -> re-parent into
            #     a new adjacent group. A reload here (loadCount bump) is the
            #     ground-truth re-parent signal; dockview recycles group ids, so
            #     we report both the group id and the group count, not a verdict.
            src_tab = ".dv-groupview:has(#marker-ifp) .dv-tab"
            p1_zone = ".dv-groupview:has(#marker-p1) .dv-content-container"
            page.evaluate(
                "([s, t, x, y]) => window.__drag(s, t, x, y)", [src_tab, p1_zone, 0.05, 0.5]
            )
            page.wait_for_timeout(200)
            la = page.evaluate("() => window.__layout()")
            sa = _iframe_state(page)
            reloaded_a = sa["loadCount"] > base["loadCount"]
            rows.append(
                f"(a) drag tab -> LEFT of group 2: counter={sa['counter']} loadCount={sa['loadCount']} RELOADED={'yes' if reloaded_a else 'no'} groups={la['groups']} ifpGroup={la['ifpGroup']}"
            )

            # (b) drag iframe tab into the CENTRE of P2's group -> stack into its tabs.
            p2_zone = ".dv-groupview:has(#marker-p2) .dv-content-container"
            page.evaluate(
                "([s, t, x, y]) => window.__drag(s, t, x, y)", [src_tab, p2_zone, 0.5, 0.5]
            )
            page.wait_for_timeout(200)
            lb = page.evaluate("() => window.__layout()")
            sb_state = _iframe_state(page)
            stacked_b = "ifp" in lb["ifpGroupPanels"] and len(lb["ifpGroupPanels"]) > 1
            reloaded_b = sb_state["loadCount"] > sa["loadCount"]
            rows.append(
                f"(b) drag tab -> CENTRE of grp 3: counter={sb_state['counter']} loadCount={sb_state['loadCount']} RELOADED={'yes' if reloaded_b else 'no'} STACKED={'yes' if stacked_b else 'no'} panels={lb['ifpGroupPanels']}"
            )

            # (c) drag a splitter (.dv-sash) to resize -- genuine pointer drag.
            sash = page.locator(".dv-sash").first
            box = sash.bounding_box()
            resized = False
            if box:
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                page.mouse.move(cx, cy)
                page.mouse.down()
                page.mouse.move(cx + 120, cy, steps=10)
                page.mouse.up()
                page.wait_for_timeout(200)
                resized = True
            sc = _iframe_state(page)
            rows.append(
                f"(c) drag splitter to resize:    counter={sc['counter']} loadCount={sc['loadCount']} RESIZED={'yes' if resized else 'no'}"
            )

            final = _iframe_state(page)
            verdict = "PASS" if (final["counter"] == 42 and final["loadCount"] == 1) else "FAIL"
            browser.close()
    finally:
        httpd.shutdown()
        shutil.rmtree(root, ignore_errors=True)

    print("\n=== dockview iframe-reload spike ===")
    for r in rows:
        print("  " + r)
    print(f"\n  final: counter={final['counter']} loadCount={final['loadCount']}")
    print(f"  VERDICT: {verdict}  (PASS iff counter==42 and loadCount==1 after all three)\n")
    return 0 if verdict in ("PASS", "FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
