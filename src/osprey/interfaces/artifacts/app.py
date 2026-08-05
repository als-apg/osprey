"""OSPREY Artifact Gallery — FastAPI Application.

A unified gallery for interactive artifacts (plots, tables, HTML, markdown)
produced by Claude during analysis sessions.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from itertools import chain
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.vendor import vendor_url
from osprey.utils.timeseries import (
    downsample_channel_map,
    extract_channel_series,
)

STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url

# Snippet injected into Plotly/table/generic HTML artifacts so they fill the
# iframe viewport in Focus Mode.  CSS alone is not enough for Plotly because
# Plotly.newPlot() applies layout.width/height via JS *after* load, overriding
# CSS.  We therefore inject a script that deletes those fixed dimensions and
# calls Plotly.Plots.resize() once the library is ready.
_RESPONSIVE_PLOTLY = r"""<style>
/* OSPREY: fill iframe viewport */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
.plotly-graph-div { width: 100% !important; height: 100vh !important; }
.js-plotly-plot { width: 100% !important; height: 100vh !important; visibility: hidden; }
table { max-width: 100%; }
</style>
<script>
/* OSPREY: responsive sizing + theme-aware backgrounds
 *
 * Anti-flash strategy: Plotly renders with its original colors before we can
 * re-theme it.  We hide charts via CSS (visibility:hidden) and only reveal
 * them after Plotly.relayout() applies the correct theme.  The page background
 * is set immediately from a <style> injected before any body content paints.
 */
(function(){
  var THEMES = {
    dark: {
      paper_bgcolor: '#131c2e', plot_bgcolor: '#0b1120',
      font: { color: '#8b9ab5' },
      xaxis: { gridcolor: 'rgba(100,116,139,0.1)', linecolor: 'rgba(100,116,139,0.18)' },
      yaxis: { gridcolor: 'rgba(100,116,139,0.1)', linecolor: 'rgba(100,116,139,0.18)' },
      legend: { bgcolor: 'rgba(19,28,46,0.85)', bordercolor: 'rgba(100,116,139,0.18)' },
      scene: { bgcolor: '#0b1120', axis_bg: '#131c2e',
               gridcolor: 'rgba(100,116,139,0.15)', spikecolor: 'rgba(100,116,139,0.4)' }
    },
    light: {
      paper_bgcolor: '#f7f9fc', plot_bgcolor: '#f7f9fc',
      font: { color: '#0c1322' },
      xaxis: { gridcolor: 'rgba(0,0,0,0.08)', linecolor: 'rgba(0,0,0,0.12)' },
      yaxis: { gridcolor: 'rgba(0,0,0,0.08)', linecolor: 'rgba(0,0,0,0.12)' },
      legend: { bgcolor: 'rgba(247,249,252,0.9)', bordercolor: 'rgba(0,0,0,0.1)' },
      scene: { bgcolor: '#f7f9fc', axis_bg: '#eef2f7',
               gridcolor: 'rgba(0,0,0,0.1)', spikecolor: 'rgba(0,0,0,0.2)' }
    }
  };

  function detectTheme() {
    try { return window.parent.document.documentElement.getAttribute('data-theme') || 'dark'; }
    catch(e) { return 'dark'; }
  }

  /* Set page background immediately (runs in <head>, before body paints) */
  var _bgTheme = THEMES[detectTheme()] || THEMES.dark;
  var _bgStyle = document.createElement('style');
  _bgStyle.textContent = 'html, body { background: ' + _bgTheme.paper_bgcolor + ' !important; }';
  document.head.appendChild(_bgStyle);

  function revealCharts() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
      gd.style.visibility = 'visible';
    });
  }

  function applyTheme(theme) {
    var t = THEMES[theme] || THEMES.dark;
    _bgStyle.textContent = 'html, body { background: ' + t.paper_bgcolor + ' !important; }';
    if (document.body) document.body.style.background = t.paper_bgcolor;
    if (typeof Plotly === 'undefined') { revealCharts(); return; }
    var plots = document.querySelectorAll('.js-plotly-plot');
    if (!plots.length) { return; }
    var pending = plots.length;
    plots.forEach(function(gd) {
      var update = {
        paper_bgcolor: t.paper_bgcolor, plot_bgcolor: t.plot_bgcolor,
        'font.color': t.font.color,
        'xaxis.gridcolor': t.xaxis.gridcolor, 'xaxis.linecolor': t.xaxis.linecolor,
        'yaxis.gridcolor': t.yaxis.gridcolor, 'yaxis.linecolor': t.yaxis.linecolor,
        'legend.bgcolor': t.legend.bgcolor, 'legend.bordercolor': t.legend.bordercolor
      };
      /* 3D scenes: theme the box, axis planes, grids, and spike lines */
      if (gd.layout) {
        Object.keys(gd.layout).forEach(function(key) {
          if (key === 'scene' || /^scene\d+$/.test(key)) {
            var s = t.scene, p = key + '.';
            update[p + 'bgcolor'] = s.bgcolor;
            ['xaxis','yaxis','zaxis'].forEach(function(ax) {
              update[p + ax + '.backgroundcolor'] = s.axis_bg;
              update[p + ax + '.gridcolor'] = s.gridcolor;
              update[p + ax + '.color'] = t.font.color;
              update[p + ax + '.spikecolor'] = s.spikecolor;
            });
          }
        });
      }
      try {
        Plotly.relayout(gd, update).then(function() {
          gd.style.visibility = 'visible';
        }).catch(function() {
          gd.style.visibility = 'visible';
        });
      } catch(e) {
        gd.style.visibility = 'visible';
      }
    });
  }

  function resizeAll() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
      if (gd.layout) { delete gd.layout.width; delete gd.layout.height; }
      if (typeof Plotly !== 'undefined') { Plotly.Plots.resize(gd); }
    });
  }

  function initAll() {
    resizeAll();
    applyTheme(detectTheme());
    /* Safety net: if relayout somehow fails to reveal, force-show after 400ms */
    setTimeout(revealCharts, 400);
  }

  if (document.readyState === 'complete') { initAll(); }
  else { window.addEventListener('load', initAll); }
  window.addEventListener('resize', resizeAll);
  window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'osprey-theme-change' && e.data.theme) {
      applyTheme(e.data.theme);
    }
  });
  // Also observe parent document's data-theme attribute directly (bypasses
  // postMessage chain which can be unreliable across nested iframes)
  try {
    var parentRoot = window.parent.document.documentElement;
    new MutationObserver(function() {
      applyTheme(parentRoot.getAttribute('data-theme') || 'dark');
    }).observe(parentRoot, { attributes: true, attributeFilter: ['data-theme'] });
  } catch(e) {}
})();
</script>"""

_RESPONSIVE_TABLE_HTML = """<style>
/* OSPREY: fill iframe viewport */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: auto; }
table { max-width: 100%; }
</style>"""

# JupyterLab-style nbconvert uses <body class="jp-Notebook"> and .jp-Cell,
# NOT the classic #notebook-container.
_NOTEBOOK_RESPONSIVE_CSS = """<style>
/* OSPREY: make notebook fill iframe viewport without horizontal overflow.
 * nbconvert's JupyterLab CSS has many nested elements with padding/margin
 * that can push total width past 100%, so we apply a universal box-sizing
 * reset and suppress horizontal scroll at the body level.  Individual code
 * cells and output areas retain their own overflow-x: auto for wide content.
 */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; width: 100%; height: 100%; }
body.jp-Notebook { padding: 0 16px; overflow-x: hidden; overflow-y: auto; }
.jp-Cell { max-width: 100%; }
/* Classic nbconvert fallback */
#notebook-container, .container { max-width: 100% !important; width: 100% !important; padding: 0 16px; }
</style>"""

_RESPONSIVE_SNIPPETS = {
    "plot_html": _RESPONSIVE_PLOTLY,
    "table_html": _RESPONSIVE_TABLE_HTML,
    "html": _RESPONSIVE_TABLE_HTML,
    "dashboard_html": _RESPONSIVE_TABLE_HTML,  # Bokeh handles its own JS sizing
}

# Standalone HTML page for server-side rendered markdown.
# Embeds raw markdown as JSON inside a non-executable <script> tag, then
# renders client-side using the same marked + hljs + KaTeX pipeline as gallery.js.
_MARKDOWN_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
<script type="module">
// Follower role: this standalone page is never the theme hub (web-terminal
// is) -- it just applies whatever theme-boot.js already resolved pre-paint
// and whatever the hub broadcasts when this page is embedded as a gallery
// preview iframe. No base.css here (unlike the gallery shell): this page is
// a plain scrolling document, not a fixed-viewport SPA, and base.css's
// `overflow: hidden` would clip it -- tokens.css carries no layout rules,
// so it's safe on its own.
import {{ initTheme }} from '/design-system/js/theme-manager.js';
initTheme({{ role: 'follower' }});
</script>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link id="hljs-theme" rel="stylesheet"
      href="{hljs_dark_href}"
      data-href-dark="{hljs_dark_href}"
      data-href-light="{hljs_light_href}">
<script src="/static/js/vendor/highlight.min.js"></script>
<script src="/static/js/vendor/marked.min.js"></script>
<link rel="stylesheet" href="/static/css/vendor/katex.min.css">
<script src="/static/js/vendor/katex.min.js"></script>
<style>
body {{
  margin: 0; padding: 24px 32px;
  font-family: var(--font-display);
  background: var(--bg-primary); color: var(--text-primary);
}}
.osprey-md-rendered {{
  font-size: 14px; line-height: 1.7; max-width: 860px; margin: 0 auto;
}}
.osprey-md-rendered h1,.osprey-md-rendered h2,.osprey-md-rendered h3 {{
  margin-top: 1.2em;
}}
.osprey-md-rendered pre,.osprey-md-rendered code {{
  background: var(--neutral-tint-08); border-radius: 3px; padding: 2px 4px; font-size: 12px;
}}
.osprey-md-rendered pre {{ padding: 12px; overflow-x: auto; }}
.osprey-md-rendered pre code {{ padding: 0; background: transparent; }}
.osprey-md-rendered table {{ border-collapse: collapse; width: 100%; }}
.osprey-md-rendered th,.osprey-md-rendered td {{
  border: 1px solid var(--border-default); padding: 6px 10px;
}}
.osprey-md-rendered blockquote {{
  border-left: 3px solid var(--border-default); margin: 1em 0; padding: 0.5em 1em;
  color: var(--text-secondary);
}}
.osprey-md-rendered img {{ max-width: 100%; height: auto; }}
/* Printed output stays light-on-white regardless of the active theme,
   matching the fleet-wide "@media print never goes dark" rule -- these
   hardcoded literals are intentional, not migration debt.
   hygiene-allow-color-start */
@media print {{
  body {{ padding: 12px; background: #fff; color: #111; }}
  .osprey-md-rendered pre,.osprey-md-rendered code {{ background: #f5f5f5; }}
  .osprey-md-rendered th,.osprey-md-rendered td {{ border-color: #ddd; }}
  .osprey-md-rendered blockquote {{ border-left-color: #ddd; color: #555; }}
}}
/* hygiene-allow-color-end */
</style>
</head>
<body>
<script type="application/json" id="md-source">{md_json}</script>
<div class="osprey-md-rendered" id="md-rendered"></div>
<script>
// Renders markdown from the embedded JSON source using marked + hljs + KaTeX.
// Content originates from trusted local artifact files; marked.parse() and
// katex.renderToString() both produce sanitized HTML output.
(function() {{
  var esc = function(s) {{
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }};
  var renderer = {{
    code: function(args) {{
      var src = args.text || '';
      var lang = args.lang || '';
      var highlighted = esc(src);
      if (typeof hljs !== 'undefined' && src) {{
        try {{
          if (lang && hljs.getLanguage(lang)) {{
            highlighted = hljs.highlight(src, {{ language: lang }}).value;
          }} else {{
            highlighted = hljs.highlightAuto(src).value;
          }}
        }} catch(e) {{}}
      }}
      return '<pre><code class="hljs' + (lang ? ' language-' + lang : '') +
             '">' + highlighted + '</code></pre>';
    }}
  }};
  marked.use({{ gfm: true, breaks: false, renderer: renderer }});

  function renderMath(text) {{
    if (typeof katex === 'undefined') return marked.parse(text);
    var placeholders = [], idx = 0;
    function ph(html) {{
      var key = '\\x00MATH' + (idx++) + '\\x00';
      placeholders.push({{ key: key, html: html }});
      return key;
    }}
    function rk(expr, dm) {{
      try {{
        return katex.renderToString(expr.trim(), {{
          displayMode: dm, throwOnError: false, strict: false
        }});
      }} catch(e) {{
        return '<span class="katex-error">' + esc(expr) + '</span>';
      }}
    }}
    text = text.replace(/\\$\\$([\\s\\S]+?)\\$\\$/g, function(_, e) {{ return ph(rk(e, true)); }});
    text = text.replace(/(?<!\\$)(?<!\\d)\\$(?!\\$)(.+?)(?<!\\$)\\$(?!\\d)/g,
      function(_, e) {{ return ph(rk(e, false)); }});
    var html;
    try {{ html = marked.parse(text); }}
    catch(e) {{ html = '<p>' + esc(text) + '</p>'; }}
    for (var i = 0; i < placeholders.length; i++) {{
      html = html.replace(placeholders[i].key, placeholders[i].html);
    }}
    return html;
  }}

  var src = JSON.parse(document.getElementById('md-source').textContent);
  // Safe: marked.parse() and katex.renderToString() produce sanitized HTML
  // from trusted local artifact content (not user input from the web).
  document.getElementById('md-rendered').innerHTML = renderMath(src);  // trusted content
}})();
</script>
</body>
</html>"""


def _build_markdown_page(md_source: str, title: str) -> str:
    """Build a standalone HTML page that renders markdown client-side."""
    # Escape for safe embedding inside <script type="application/json">
    md_json = json.dumps(md_source).replace("</", r"<\/")
    return _MARKDOWN_PAGE_TEMPLATE.format(
        title=title.replace("&", "&amp;").replace("<", "&lt;"),
        md_json=md_json,
        hljs_dark_href=vendor_url(
            "highlight.js atom-one-dark theme", "/static/css/vendor/atom-one-dark.min.css"
        ),
        hljs_light_href=vendor_url(
            "highlight.js atom-one-light theme", "/static/css/vendor/atom-one-light.min.css"
        ),
    )


_CDN_PLOTLY_RE = re.compile(r'(src=["\'])https://cdn\.plot\.ly/plotly[^"\']*\.min\.js(["\'])')

# Strip SRI attributes — the local copy may differ from the CDN version.
_SRI_ATTR_RE = re.compile(r'\s+(?:integrity|crossorigin)=["\'][^"\']*["\']')


def _rewrite_plotly_cdn(html_bytes: bytes) -> bytes:
    """In offline mode, replace CDN Plotly URLs with the local bundled copy.

    Also strips ``integrity`` and ``crossorigin`` attributes from the same
    ``<script>`` tag, since the local file may differ from the CDN version
    and SRI would block execution.

    In default (CDN) mode this is a no-op — the browser fetches plotly
    directly from ``cdn.plot.ly`` with its original SRI attributes intact.
    """
    from osprey.interfaces.vendor import is_offline

    if not is_offline():
        return html_bytes
    html = html_bytes.decode("utf-8", errors="replace")
    if "cdn.plot.ly/plotly" not in html:
        return html_bytes
    html = _CDN_PLOTLY_RE.sub(r"\1/static/js/vendor/plotly-3.3.1.min.js\2", html)
    html = _SRI_ATTR_RE.sub("", html)
    return html.encode("utf-8")


def _inject_html_snippet(html_bytes: bytes, snippet: str) -> bytes:
    """Inject an HTML snippet (CSS/JS) into HTML content, before </head>."""
    html = html_bytes.decode("utf-8", errors="replace")
    if "</head>" in html:
        html = html.replace("</head>", snippet + "\n</head>", 1)
    elif "</body>" in html:
        html = html.replace("</body>", snippet + "\n</body>", 1)
    else:
        html = snippet + html
    return html.encode("utf-8")


class FocusRequest(BaseModel):
    artifact_id: str
    fullscreen: bool = False


class PinRequest(BaseModel):
    pinned: bool = True


class _SSEBroadcaster:
    """Manages per-client asyncio.Queue instances for SSE push."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[dict]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def broadcast(self, data: dict) -> None:
        """Push data to all connected SSE clients (called from sync context)."""
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass  # Drop if client is too slow


MAX_TIMESERIES_FILE_BYTES = 200 * 1024 * 1024  # 200 MB


def _union_timestamp_axis(series: dict[str, dict]) -> Collection:
    """Every timestamp any channel carries, deduplicated — the table's row axis.

    Shared by both response formats so ``format=chart``'s ``summary.row_count``
    is by construction the same number ``format=table`` paginates over
    (``total_rows``). They are two separate requests whose numbers an operator
    reads side by side in the same view, so they must not be computed twice.
    """
    stamps = chain.from_iterable(data.get("timestamps", []) for data in series.values())
    try:
        return set(stamps)
    except TypeError:
        # An unhashable timestamp (a JSON array, say) cannot go in a set. Same
        # reasoning as the incomparable-types fallback in the sort below: this
        # endpoint renders whatever JSON an entry's `data_file` points at,
        # including artifacts written by hand or by another facility's script,
        # and such a file is still displayable. Dedupe by equality instead --
        # quadratic, but only on this malformed-artifact path, and a slow render
        # beats a 500. `stamps` is a one-shot iterator, so rebuild it.
        unique: list = []
        for stamp in chain.from_iterable(data.get("timestamps", []) for data in series.values()):
            if stamp not in unique:
                unique.append(stamp)
        return unique


def _raise_duplicate_sample(channel: str, stamp: object) -> None:
    """Reject two samples sharing one (timestamp, channel) cell.

    Extracted so the hashable and unhashable lookups cannot drift into
    reporting the same malformed artifact differently — the wording and the
    choice of *which* repeat is named are both part of this endpoint's
    contract, pinned by tests.
    """
    raise HTTPException(
        status_code=500,
        detail=(
            f"Channel {channel!r} has more than one sample at timestamp {stamp!r}; "
            "a table view has exactly one cell per channel per timestamp "
            "and cannot represent both without silently discarding one."
        ),
    )


class _EqualityMatchLookup:
    """A channel's ``timestamp -> value`` lookup for unhashable timestamps.

    Stands in for the ``dict`` the fast path builds, matching by ``==`` rather
    than by hash so a timestamp that is itself a JSON array still resolves. Only
    a malformed artifact reaches this, so the linear scan per cell — and the
    quadratic duplicate check below — are paid only there; the hashable path is
    untouched.

    Duplicates are detected eagerly, in the constructor, so that a repeated
    label fails exactly as loudly here as it does on the hashable path instead
    of silently resolving to whichever sample the scan happens to reach first.
    """

    __slots__ = ("_pairs",)

    def __init__(self, channel: str, timestamps: list, values: list) -> None:
        # `strict=False` mirrors the fast path: a channel whose timestamps and
        # values differ in length is tolerated, not an error.
        self._pairs = list(zip(timestamps, values, strict=False))
        for i, (stamp, _val) in enumerate(self._pairs):
            if any(stamp == earlier for earlier, _ in self._pairs[:i]):
                _raise_duplicate_sample(channel, stamp)

    def get(self, key: object, default: object = None) -> object:
        for stamp, value in self._pairs:
            if stamp == key:
                return value
        return default


def _pivot_channel_series_to_table(
    series: dict[str, dict],
    *,
    offset: int,
    limit: int,
) -> tuple[list[str], list, list[list], int]:
    """Pivot per-channel series into aligned rows for table display.

    Channels are not aligned to a shared axis (see ``extract_channel_series``),
    but the table view still needs one row per timestamp with one cell per
    channel. This unions every channel's own timestamps into one sorted axis
    and looks up each channel's value at each shared timestamp, leaving
    ``None`` where that channel has no sample there.

    This is presentation-only: a no-fill pivot computed fresh on every
    request, never written back to storage, and independent of the
    per-channel ``series`` a caller may also use for the chart path.

    Two malformed-input cases are handled deliberately, not by accident:

    - A channel with more than one sample at the *same* timestamp label
      cannot be represented here -- a table has exactly one cell per
      (timestamp, channel) pair. Silently keeping the last sample would
      quietly discard the other(s), so this raises a clear error instead
      (a visible failure beats a corrupted row count).
    - A channel whose ``timestamps``/``values`` lengths differ (a malformed
      artifact, not a shape collision) is tolerated the same way the chart
      path already tolerates it -- pairs are built with ``zip(..., strict=
      False)``, so the mismatch never crashes the endpoint.

    Only the requested page's rows are materialized. Building every row and
    slicing afterwards costs one cell per (timestamp, channel) pair across the
    whole file — on every page click, against a file cap of
    ``MAX_TIMESERIES_FILE_BYTES`` and a default page of 50 rows. The shared
    axis still has to be unioned and sorted in full to know the row count and
    where the page starts, but that is one entry per timestamp rather than one
    per timestamp *and* channel.

    The column order is the series' own channel order, derived here rather than
    supplied: the returned ``columns`` is the very list each row was indexed
    with, so a caller cannot pair these rows with a header from anywhere else
    (the table view's header used to come from a *separate* ``format=chart``
    request, which can disagree for an artifact being written while it is read).

    Args:
        series: Per-channel series as returned by ``extract_channel_series``.
        offset: Index of the first row to return.
        limit: Maximum rows to return.

    Returns:
        Tuple of (columns, index, data, total_rows) -- the channel names in
        column order, the requested page's timestamps, its rows with one value
        (or ``None``) per column, and the full row count the page was taken
        from.

    Raises:
        HTTPException: if any channel has more than one sample at the same
            timestamp label.
    """
    columns = list(series.keys())
    # Either lookup answers `.get(timestamp)`; which one a channel gets depends
    # only on whether its timestamps are hashable.
    value_by_channel: dict[str, dict | _EqualityMatchLookup] = {}
    for ch, data in series.items():
        timestamps = data.get("timestamps", [])
        values = data.get("values", [])
        # A dict of n pairs holds fewer than n entries iff a key repeated, so
        # the length check detects duplicates exactly, while the build itself
        # stays in C. Only malformed input pays for the re-walk below.
        try:
            by_ts: dict = dict(zip(timestamps, values, strict=False))
        except TypeError:
            # An unhashable timestamp (a JSON array, say) cannot key a dict at
            # all. Same reasoning as the incomparable-types sort fallback below:
            # the cells are still displayable, so match by equality instead of
            # by hash rather than 500 on a renderable artifact.
            value_by_channel[ch] = _EqualityMatchLookup(ch, timestamps, values)
            continue
        if len(by_ts) != min(len(timestamps), len(values)):
            # Which duplicate is named is part of the error's contract: the
            # first repeat reached scanning pairs left to right, not the first
            # label that happens to repeat somewhere later. Re-walking the same
            # zip the fast path collapsed is what keeps those identical.
            seen: set = set()
            for ts, _val in zip(timestamps, values, strict=False):
                if ts in seen:
                    _raise_duplicate_sample(ch, ts)
                seen.add(ts)
        value_by_channel[ch] = by_ts

    all_timestamps = _union_timestamp_axis(series)
    try:
        index = sorted(all_timestamps)
    except TypeError:
        # Timestamps of mutually-incomparable types (e.g. int from one channel,
        # str from another) can't be compared by `<`. No writer in this repo
        # produces that -- but this endpoint renders whatever JSON an entry's
        # `data_file` points at, including artifacts written by hand or by
        # another facility's script, and every value in such a file is
        # displayable. A deterministic (if not necessarily chronological) order
        # beats a 500 on data that is otherwise perfectly renderable; contrast
        # the duplicate-timestamp branch above, where the alternative to
        # failing is silently dropping a real sample.
        index = sorted(all_timestamps, key=str)

    total_rows = len(index)
    page = index[offset : min(offset + limit, total_rows)]
    rows = [[value_by_channel[ch].get(ts) for ch in columns] for ts in page]
    return columns, page, rows, total_rows


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Artifact Gallery FastAPI application.

    Args:
        workspace_root: Workspace root containing ``artifacts/`` dir.
            Defaults to ``./_agent_data``.
    """
    from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher
    from osprey.stores.artifact_store import (
        ArtifactEntry,
        ArtifactStore,
        register_artifact_delete_listener,
        register_artifact_listener,
        unregister_artifact_delete_listener,
        unregister_artifact_listener,
    )

    store = ArtifactStore(workspace_root=workspace_root)

    # Prime config and load custom artifact categories (if available)
    try:
        config_path = (workspace_root or Path("_agent_data")) / "config.yml"
        if config_path.exists():
            from osprey.utils.config import get_config_builder

            get_config_builder(config_path=str(config_path), set_as_default=True)
            from osprey.stores.type_registry import load_categories_from_config

            load_categories_from_config()
    except Exception:
        pass  # Config may not be available in all contexts

    broadcaster = _SSEBroadcaster()

    index_watcher = StoreIndexWatcher(
        workspace_root=workspace_root,
        broadcaster=broadcaster,
        artifact_store=store,
    )

    def _on_artifact_saved(entry: ArtifactEntry) -> None:
        broadcaster.broadcast({"type": "artifact", **entry.to_dict()})

    def _on_artifact_deleted(entry: ArtifactEntry) -> None:
        if app.state.focused_artifact_id == entry.id:
            app.state.focused_artifact_id = None
        _write_focus_file()
        broadcaster.broadcast({"type": "artifact_deleted", "id": entry.id})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        register_artifact_listener(_on_artifact_saved)
        register_artifact_delete_listener(_on_artifact_deleted)
        index_watcher.start()
        yield
        index_watcher.stop()
        unregister_artifact_delete_listener(_on_artifact_deleted)
        unregister_artifact_listener(_on_artifact_saved)

    app = FastAPI(
        title="OSPREY Artifact Gallery",
        description="Interactive gallery for analysis artifacts",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.state.artifact_store = store
    app.state.focused_artifact_id = None  # None = show latest

    focus_file = workspace_root / "focus_state.txt"

    def _write_focus_file() -> None:
        """Write current focus state to a plain-text file for the CLI hook."""
        lines: list[str] = []
        aid = app.state.focused_artifact_id
        if aid:
            entry = store.get_entry(aid)
            if entry:
                lines.append(f'  artifact: "{entry.title}" (id={aid})')
        # List pinned artifacts
        pinned = store.list_entries(pinned=True)
        for p in pinned:
            if p.id != aid:
                lines.append(f'  pinned:   "{p.title}" (id={p.id})')
        if lines:
            focus_file.write_text("[Gallery Focus]\n" + "\n".join(lines) + "\n")
        elif focus_file.exists():
            focus_file.write_text("")

    # --- Routes ---

    @app.get("/")
    async def root(request: Request):
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/health")
    async def health():
        return {"status": "healthy", "artifact_count": len(store.list_entries())}

    @app.get("/api/type-registry")
    async def get_type_registry():
        from osprey.stores.type_registry import registry_to_api_dict

        return JSONResponse(registry_to_api_dict())

    @app.get("/api/events")
    async def sse_events():
        q = broadcaster.subscribe()

        async def stream():
            try:
                while True:
                    data = await q.get()
                    yield f"data: {json.dumps(data)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # --- Artifact routes ---

    @app.get("/api/artifacts")
    async def list_artifacts(
        type: str | None = None,
        search: str | None = None,
        pinned: bool | None = Query(None),
        category: str | None = None,
        session_id: str | None = None,
    ):
        entries = store.list_entries(
            type_filter=type,
            search=search,
            pinned=pinned,
            category_filter=category,
            session_filter=session_id,
        )
        return {
            "count": len(entries),
            "artifacts": [e.to_dict() for e in entries],
        }

    @app.get("/api/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        return entry.to_dict()

    @app.post("/api/artifacts/{artifact_id}/pin")
    async def pin_artifact(artifact_id: str, req: PinRequest):
        entry = store.set_pinned(artifact_id, req.pinned)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        _write_focus_file()
        broadcaster.broadcast({"type": "artifact_updated", **entry.to_dict()})
        return {"status": "ok", "artifact_id": artifact_id, "pinned": entry.pinned}

    @app.get("/api/artifacts/{artifact_id}/data")
    async def get_artifact_data(
        artifact_id: str,
        format: str | None = Query(None, pattern="^(chart|table)$"),
        max_points: int = Query(2000, ge=10, le=50000),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=10000),
    ):
        """Serve timeseries data for artifacts with metadata.data_file."""
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

        data_file = entry.data_file or entry.metadata.get("data_file")
        if not data_file:
            raise HTTPException(status_code=400, detail="Artifact has no associated data file")

        filepath = Path(data_file)
        if not filepath.is_absolute():
            # data_file may be (a) a project-CWD-relative path like
            # "_agent_data/artifacts/foo.json" (current ArtifactStore format),
            # (b) a bare filename (legacy entries written before the format
            # change), or (c) some other workspace-relative path. Try each
            # candidate; the legacy DataContext path used absolute strings
            # which are handled by the is_absolute() branch above.
            candidates = [
                store._workspace.parent / filepath,
                store._store_dir / filepath,
                store._workspace / filepath,
            ]
            for candidate in candidates:
                if candidate.exists():
                    filepath = candidate
                    break
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Data file not found on disk")

        # No format param → return full file as-is
        if format is None:
            return Response(content=filepath.read_bytes(), media_type="application/json")

        # format=chart or format=table requires timeseries data
        data_type = entry.metadata.get("data_type", "")
        if data_type != "timeseries" and entry.category != "archiver_data":
            raise HTTPException(
                status_code=400,
                detail="format parameter is only supported for timeseries data",
            )

        file_size = filepath.stat().st_size
        if file_size > MAX_TIMESERIES_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({file_size // (1024 * 1024)}MB). "
                    "Access the data file directly."
                ),
            )

        raw = json.loads(filepath.read_bytes())
        series, query_meta = extract_channel_series(raw)

        if format == "chart":
            channels_out = [
                {
                    "channel": record["channel"],
                    "timestamps": record["timestamps"],
                    "values": record["values"],
                    "total_points": record["original_points"],
                    "returned_points": record["returned_points"],
                    "numeric": record["numeric"],
                }
                for record in downsample_channel_map(series, max_points)
            ]
            # Cross-channel aggregates belong to whoever knows all the
            # channels. Every client would otherwise re-derive the three sums
            # itself, and none of them can derive `row_count` at all: the
            # unioned timestamp axis only exists on this side, so a client
            # showing a per-channel point sum next to the table's `total_rows`
            # shows two numbers that cannot be reconciled.
            return {
                "channels": channels_out,
                "metadata": query_meta,
                "summary": {
                    "total_points": sum(ch["total_points"] for ch in channels_out),
                    "returned_points": sum(ch["returned_points"] for ch in channels_out),
                    # Whether a channel was reduced is exactly "it came back
                    # shorter", so it is derived here rather than published
                    # per-channel as a second field the client would have to
                    # keep consistent with the two counts beside it.
                    "downsampled": any(
                        ch["returned_points"] < ch["total_points"] for ch in channels_out
                    ),
                    "row_count": len(_union_timestamp_axis(series)),
                },
            }

        # format == "table": union every channel's own timestamps into one
        # sorted axis and pivot to columns/index/data for display.
        # Presentation-only -- a missing sample at a shared timestamp becomes
        # `null`, never filled, and nothing here is written back to storage.
        columns, sliced_index, sliced_data, total_rows = _pivot_channel_series_to_table(
            series, offset=offset, limit=limit
        )
        return {
            "columns": columns,
            "index": sliced_index,
            "data": sliced_data,
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "returned_rows": len(sliced_index),
        }

    @app.get("/api/focus")
    async def get_focus():
        focused_id = app.state.focused_artifact_id
        if focused_id:
            entry = store.get_entry(focused_id)
            if entry:
                return {"focused": True, "artifact": entry.to_dict()}
            # Stale focus — clear it and fall back to latest
            app.state.focused_artifact_id = None

        # Fall back to latest artifact
        entries = store.list_entries()
        if entries:
            return {"focused": False, "artifact": entries[-1].to_dict()}
        return {"focused": False, "artifact": None}

    @app.post("/api/focus")
    async def set_focus(req: FocusRequest):
        entry = store.get_entry(req.artifact_id)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {req.artifact_id} not found",
            )
        app.state.focused_artifact_id = req.artifact_id
        _write_focus_file()
        event = {"type": "focus", "domain": "artifact", "id": req.artifact_id}
        if req.fullscreen:
            event["fullscreen"] = True
        broadcaster.broadcast(event)
        return {"status": "ok", "artifact_id": req.artifact_id}

    @app.get("/files/{artifact_id}/{filename}")
    async def serve_file(artifact_id: str, filename: str):
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

        filepath = store.get_file_path(artifact_id)
        if not filepath or not filepath.exists():
            raise HTTPException(status_code=404, detail="Artifact file not found on disk")

        # For binary files (images), use FileResponse for proper streaming
        snippet = _RESPONSIVE_SNIPPETS.get(entry.artifact_type)
        if not snippet:
            # Text artifacts (e.g. .tex with application/x-tex) wouldn't render
            # inline in an iframe with their original non-browser MIME type —
            # browsers trigger a download instead. Serve as text/plain so the
            # gallery preview iframe shows the source.
            media_type = (
                "text/plain; charset=utf-8" if entry.artifact_type == "text" else entry.mime_type
            )
            return FileResponse(
                filepath,
                media_type=media_type,
                filename=entry.filename,
                content_disposition_type="inline",
            )

        # HTML types may need responsive snippet injection + CDN rewriting.
        content = filepath.read_bytes()
        # Always rewrite CDN Plotly URLs to local — artifacts may have been
        # generated with include_plotlyjs='cdn' regardless of what OSPREY's
        # own code paths use, and the CDN is unreachable in offline deployments.
        content = _rewrite_plotly_cdn(content)
        if entry.artifact_type == "plot_html":
            # Only inject the local Plotly bundle if the HTML doesn't already
            # have one (e.g. include_plotlyjs=False). Avoid duplicates — the
            # 4.8MB file takes ~1s through the reverse proxy per load.
            if b"plotly-3.3.1.min.js" not in content:
                plotly_src = vendor_url("Plotly.js", "/static/js/vendor/plotly-3.3.1.min.js")
                snippet = f'<script src="{plotly_src}"></script>\n' + snippet
        content = _inject_html_snippet(content, snippet)
        return Response(
            content=content,
            media_type=entry.mime_type,
            headers={"Content-Disposition": f'inline; filename="{entry.filename}"'},
        )

    @app.delete("/api/artifacts/{artifact_id}")
    async def delete_artifact(artifact_id: str):
        deleted = store.delete_entry(artifact_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        return {"status": "ok", "artifact_id": artifact_id}

    @app.get("/api/notebooks/{artifact_id}/rendered")
    async def render_notebook(artifact_id: str):
        """Render a notebook artifact to HTML on-the-fly with caching."""
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        if entry.artifact_type != "notebook":
            raise HTTPException(status_code=400, detail="Artifact is not a notebook")

        filepath = store.get_file_path(artifact_id)
        if not filepath or not filepath.exists():
            raise HTTPException(status_code=404, detail="Notebook file not found on disk")

        try:
            from osprey.stores.notebook_renderer import get_or_render_html

            cache_dir = store.artifact_dir / "_notebook_cache"
            html, _ = get_or_render_html(filepath, cache_dir=cache_dir)
            html_bytes = _inject_html_snippet(html.encode("utf-8"), _NOTEBOOK_RESPONSIVE_CSS)
            return HTMLResponse(content=html_bytes.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Notebook rendering failed: {exc}"
            ) from exc

    @app.get("/api/markdown/{artifact_id}/rendered")
    async def render_markdown(artifact_id: str):
        """Render a markdown artifact to a standalone HTML page."""
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        if entry.artifact_type != "markdown":
            raise HTTPException(status_code=400, detail="Artifact is not a markdown file")

        filepath = store.get_file_path(artifact_id)
        if not filepath or not filepath.exists():
            raise HTTPException(status_code=404, detail="Markdown file not found on disk")

        md_source = filepath.read_text(encoding="utf-8", errors="replace")
        html = _build_markdown_page(md_source, entry.title or entry.filename or "Markdown")
        return HTMLResponse(content=html)

    @app.get("/api/notebooks/{artifact_id}/interactive")
    async def interactive_notebook(artifact_id: str):
        """Return JupyterLab URL for interactive notebook viewing."""
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        if entry.artifact_type != "notebook":
            raise HTTPException(status_code=400, detail="Artifact is not a notebook")

        filepath = store.get_file_path(artifact_id)
        if not filepath or not filepath.exists():
            raise HTTPException(status_code=404, detail="Notebook file not found")

        jupyter_path = f"artifacts/{entry.filename}"
        jupyter_url = f"http://127.0.0.1:8088/doc/tree/{jupyter_path}"

        return {
            "jupyter_url": jupyter_url,
            "artifact_id": artifact_id,
        }

    # Logbook entry composer
    from osprey.interfaces.artifacts.logbook import logbook_router

    app.include_router(logbook_router)

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8086,
    workspace_root: Path | None = None,
) -> None:
    """Run the artifact gallery server.

    Args:
        host: Host to bind to.
        port: Port to run on.
        workspace_root: Workspace root dir.
    """
    import uvicorn

    app = create_app(workspace_root=workspace_root)
    uvicorn.run(app, host=host, port=port, log_level="info")
