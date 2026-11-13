"""OSPREY Artifact Gallery — FastAPI Application.

A unified gallery for interactive artifacts (plots, tables, HTML, markdown)
produced by Claude during analysis sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import AsyncIterator, Collection, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, model_validator

from osprey.agent_runner.artifact_resolve import deployed_render_dir
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.artifacts import inline_assets
from osprey.interfaces.artifacts.content_sandbox import ContentSandboxMiddleware, not_content
from osprey.interfaces.opaque_origin import document_headers, opaque_headers
from osprey.interfaces.vendor import html_has_plotly_bundle, is_offline, vendor_url
from osprey.utils.timeseries import (
    downsample_channel_map,
    extract_channel_series,
)
from osprey_connectors.config import get_facility_timezone

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url


def _shell_vendor_script(name: str) -> str:
    """A vendor ``<script>`` for the gallery shell: CDN online, inline offline.

    For assets the manifest does not ship into this app's own ``/static``
    (DOMPurify ships to the web terminal's vendor directory), so the shell
    cannot reference a local path for them.
    """
    from markupsafe import Markup

    return Markup(
        inline_assets.vendor_script(name, offline=is_offline())
    )  # vendor file or CDN tag, never agent text


templates.env.globals["vendor_script"] = _shell_vendor_script

# Head of every injected snippet: it makes the served artifact a design-system
# page like any other.  theme-boot.js resolves `data-theme` pre-paint (?theme=
# from the gallery, then the operator's stored preference, then the
# server-rendered `web.theme` pin, then the OS) and tokens.css supplies that
# theme's colors.  Each snippet's own module script then takes the follower
# role, so the page also follows what the hub broadcasts while it is embedded
# as a preview iframe.
_DESIGN_SYSTEM_HEAD = """<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">"""

# Every page below is served into the content sandbox's opaque origin
# (``self.origin === 'null'``). Each one's module script calls theme-manager.js's
# ``initTheme({ role: 'follower' })``, which on an opaque page takes the hub's
# live ``osprey-theme-change`` broadcast that the gallery relays to its preview
# iframes from the embedding parent only.

# C-MSG: every page served on the opaque origin carries this listener.  The
# gallery cannot reach into an opaque document, so the document announces
# itself (``osprey-artifact-ready``, to whichever window opened or embeds it,
# with target "*" -- an opaque frame cannot name the console's origin and the
# message carries nothing) and prints itself when asked
# (``osprey-artifact-print``, accepted only from that opener or parent).  Theme
# messages are the design-system follower's, imported beside this listener.
_ARTIFACT_MESSAGE_LISTENER = """<script>
/* OSPREY: C-MSG ready/print listener. */
(function () {
  var peer = window.opener || window.parent;
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || data.type !== 'osprey-artifact-print') { return; }
    if (event.source !== window.opener && event.source !== window.parent) { return; }
    window.print();
  });
  function ready() {
    if (peer && peer !== window) { peer.postMessage({ type: 'osprey-artifact-ready' }, '*'); }
  }
  if (document.readyState === 'complete') { ready(); }
  else { window.addEventListener('load', ready); }
})();
</script>"""

# Snippet injected into Plotly artifacts so they fill the iframe viewport in
# Focus Mode and follow the OSPREY theme.  CSS alone is not enough for the
# sizing: Plotly.newPlot() applies layout.width/height via JS *after* load,
# overriding CSS, so the script below deletes those fixed dimensions and calls
# Plotly.Plots.resize() once the library is ready.  Every color comes from the
# --chart-* tokens via chartRelayout(), so all eight themes work and nothing
# here needs to know what "light" or "dark" looks like.  In a frame the page
# also takes the gallery's content-strip commands (zoom, pan, autoscale, reset
# axes, export PNG) from plot-commands.js, tells the gallery while the pointer
# is over it (frame-pointer.js: the strip shows then), and hides Plotly's own
# modebar, which the strip replaces; a plot opened on its own keeps the modebar.
_RESPONSIVE_PLOTLY = (
    _DESIGN_SYSTEM_HEAD
    + "\n"
    + _ARTIFACT_MESSAGE_LISTENER
    + r"""
<style>
/* OSPREY: fill iframe viewport; page background from the theme in force,
   painted before any chart -- the chart itself stays hidden until it has been
   re-themed (anti-flash: Plotly first draws the author's baked-in colors). */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
             background: var(--chart-paper-bg) !important; }
.plotly-graph-div { width: 100% !important; height: 100vh !important; }
.js-plotly-plot { width: 100% !important; height: 100vh !important; visibility: hidden; }
table { max-width: 100%; }
/* OSPREY: framed, the gallery's content strip replaces Plotly's modebar
   (plot-commands.js marks the page); a plot opened on its own keeps it. */
html[data-osprey-framed] .modebar-container,
html[data-osprey-framed] .modebar { display: none !important; }
/* OSPREY: the browser's own print of the sandboxed page omits plot chrome. */
@media print { .plotly-notifier, .modebar, .modebar-container { display: none !important; } }
</style>
<script type="module">
/* OSPREY: responsive sizing, live re-theming from the design tokens, and the
   gallery's content-strip commands. */
import { initTheme, subscribe, chartRelayout } from '/design-system/js/theme-manager.js';
import { followPlotCommands, rememberViews } from '/static/js/plot-commands.js';
import { reportPointer } from '/static/js/frame-pointer.js';

function plots() { return Array.from(document.querySelectorAll('.js-plotly-plot')); }
function reveal(gd) { gd.style.visibility = 'visible'; }

function applyTheme() {
  if (typeof Plotly === 'undefined') { plots().forEach(reveal); return; }
  plots().forEach((gd) => {
    try {
      Plotly.relayout(gd, chartRelayout(gd)).then(() => reveal(gd), () => reveal(gd));
    } catch {
      reveal(gd);
    }
  });
}

function resizeAll() {
  plots().forEach((gd) => {
    if (gd.layout) { delete gd.layout.width; delete gd.layout.height; }
    if (typeof Plotly !== 'undefined') { Plotly.Plots.resize(gd); }
  });
}

let loaded = false;
function initAll() {
  loaded = true;
  resizeAll();
  rememberViews(plots());
  applyTheme();
  /* Safety net: if relayout somehow fails to reveal, force-show after 400ms */
  setTimeout(() => plots().forEach(reveal), 400);
}

initTheme({ role: 'follower' });
followPlotCommands(window, plots);
reportPointer(window);
// Re-theme in place on every later apply (a hub broadcast, or the gallery's
// re-send when a hidden iframe becomes visible). Before `load` there are no
// charts to theme, and revealing them then would flash the author's own
// colors -- so initAll() owns the first apply.
subscribe(() => { if (loaded) applyTheme(); });

if (document.readyState === 'complete') { initAll(); }
else { window.addEventListener('load', initAll); }
window.addEventListener('resize', resizeAll);
</script>"""
)

# Injected into table_html, agent-authored html, and dashboard_html artifacts.
# Beyond viewport sizing it paints THEMED DEFAULTS at zero specificity via
# :where() -- a page that styles itself (an agent-authored report with its own
# palette) beats every one of these rules regardless of rule order, while an
# unstyled fragment (a pandas table, a bare snippet) stops rendering as a
# black-on-white browser-default island inside a themed gallery.
# ``.artifact-table`` is the class serialize_object() puts on DataFrame tables.
_RESPONSIVE_TABLE_HTML = (
    _DESIGN_SYSTEM_HEAD
    + "\n"
    + _ARTIFACT_MESSAGE_LISTENER
    + """
<style>
/* OSPREY: fill iframe viewport */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: auto; }
table { max-width: 100%; }
/* OSPREY: themed defaults -- zero specificity, author styles always win */
:where(html) { background: var(--bg-primary); }
:where(body) { color: var(--text-primary); font-family: var(--font-display); }
:where(table.artifact-table) { border-collapse: collapse; }
:where(.artifact-table th, .artifact-table td) {
  border: 1px solid var(--border-default); padding: 4px 8px;
}
:where(.artifact-table th) { background: var(--bg-secondary); }
</style>
<script type="module">
import { initTheme } from '/design-system/js/theme-manager.js';
import { reportPointer } from '/static/js/frame-pointer.js';
initTheme({ role: 'follower' });
// In a frame, the gallery shows a table's content strip while the pointer is over it.
reportPointer(window);
</script>"""
)

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
/* OSPREY: the browser's own print of the sandboxed page omits the toolbar. */
@media print { .jp-Toolbar, .jp-mod-toolbar { display: none !important; } }
</style>"""

# The notebook page follows the theme like the other artifact pages:
# notebook-theme.css re-points nbconvert's --jp-* properties at the tokens, and
# the follower script keeps data-theme current while it is embedded. The font
# sheet brings the faces those tokens name; a page without it falls back to the
# viewer's own sans-serif and monospace.
_NOTEBOOK_SNIPPET = (
    _DESIGN_SYSTEM_HEAD
    + '\n<link rel="stylesheet" href="/static/fonts/fonts.css">'
    + '\n<link rel="stylesheet" href="/static/css/notebook-theme.css">\n'
    + _NOTEBOOK_RESPONSIVE_CSS
    + "\n"
    + _ARTIFACT_MESSAGE_LISTENER
    + """
<script type="module">
import { initTheme } from '/design-system/js/theme-manager.js';
initTheme({ role: 'follower' });
</script>"""
)

_RESPONSIVE_SNIPPETS = {
    "plot_html": _RESPONSIVE_PLOTLY,
    "table_html": _RESPONSIVE_TABLE_HTML,
    "html": _RESPONSIVE_TABLE_HTML,
    # Bokeh handles its own JS sizing, and bakes its plot colors into its own
    # model -- the page around it follows the theme, the plot itself does not.
    "dashboard_html": _RESPONSIVE_TABLE_HTML,
}

# Standalone HTML page for server-side rendered markdown.
# Embeds raw markdown as JSON inside a non-executable <script> tag, then
# renders client-side using the same marked + hljs + KaTeX pipeline as gallery.js.
#
# The page is served into an opaque origin (content sandbox), where every
# server path answers 401 on a multi-user deployment. So the template below is
# written with ordinary server paths for the first-party parts and passed
# through ``inline_assets.self_contained`` (design-system scripts, stylesheets
# and modules become inline / ``data:`` URL), and the vendor libraries come
# from ``_markdown_vendor_head`` (CDN online, inline offline).
_MARKDOWN_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="/design-system/js/theme-boot.js"></script>
<link rel="stylesheet" href="/design-system/css/tokens.css">
{msg_listener}
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
{vendor_head}
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
<script type="module">
// Renders markdown from the embedded JSON source through the gallery's
// shared marked + hljs + KaTeX pipeline (md-render.js) — one algorithm for
// the preview pane and this standalone page. Module scripts defer, so the
// classic vendor <script> tags above have populated the marked/hljs/katex/
// DOMPurify globals by the time this runs. md-render.js owns the one
// HTML sink and sanitises it with DOMPurify (agent markdown is untrusted).
import {{ renderMarkdownInto }} from '/static/js/md-render.js';

const src = JSON.parse(document.getElementById('md-source').textContent);
renderMarkdownInto(document.getElementById('md-rendered'), src);
</script>
</body>
</html>"""


def _markdown_vendor_head(offline: bool) -> str:
    """highlight.js, marked, KaTeX (script and stylesheet) and DOMPurify for the markdown page."""
    return "\n".join(
        (
            inline_assets.vendor_script("highlight.js", offline=offline),
            inline_assets.vendor_script("marked", offline=offline),
            inline_assets.katex_stylesheet(offline=offline),
            inline_assets.vendor_script("KaTeX JS", offline=offline),
            inline_assets.vendor_script("DOMPurify", offline=offline),
        )
    )


_PAGE_SLOT_RE = re.compile("\x00OSPREY_SLOT_([a-z_]+)\x00")


def _build_markdown_page(md_source: str, title: str) -> str:
    """Build a standalone, self-contained HTML page that renders markdown client-side."""
    offline = is_offline()
    # The template's first-party parts are inlined while the agent-authored
    # title and source, and the vendor libraries, are still slots: the rewrite
    # must never scan (or touch) text it did not write. The slots are then
    # filled in one pass, so no filled value is ever re-scanned for a slot.
    page = inline_assets.self_contained(
        _MARKDOWN_PAGE_TEMPLATE.format(
            title="\x00OSPREY_SLOT_title\x00",
            md_json="\x00OSPREY_SLOT_md_json\x00",
            hljs_dark_href="\x00OSPREY_SLOT_hljs_dark\x00",
            hljs_light_href="\x00OSPREY_SLOT_hljs_light\x00",
            vendor_head="\x00OSPREY_SLOT_vendor_head\x00",
            msg_listener=_ARTIFACT_MESSAGE_LISTENER,
        )
    )
    slots = {
        "title": title.replace("&", "&amp;").replace("<", "&lt;"),
        # Escape for safe embedding inside <script type="application/json">
        "md_json": json.dumps(md_source).replace("</", r"<\/"),
        "hljs_dark": inline_assets.vendor_stylesheet_href(
            "highlight.js atom-one-dark theme", offline=offline
        ),
        "hljs_light": inline_assets.vendor_stylesheet_href(
            "highlight.js atom-one-light theme", offline=offline
        ),
        "vendor_head": _markdown_vendor_head(offline),
    }
    return _PAGE_SLOT_RE.sub(lambda m: slots[m.group(1)], page)


#: The opening ``<html`` tag, provided it carries no ``data-theme`` of its own.
_HTML_TAG_WITHOUT_THEME_RE = re.compile(r"<html(?![^>]*\bdata-theme=)", re.IGNORECASE)


def _stamp_data_theme(html: str, theme_id: str | None) -> str:
    """Server-render a pinned theme as ``<html data-theme="...">``.

    The same server rung the web terminal renders for its own page: a page
    served standalone (an artifact opened in its own tab, a rendered markdown
    or notebook page) has no hub to follow, and on a first visit -- nothing in
    ``localStorage``, no ``?theme=`` -- ``theme-boot.js`` would otherwise fall
    through to the OS preference and ignore a deployment's pin.

    ``theme_id`` is ``None`` when nothing is pinned, and the page is then left
    untouched. A page that already carries ``data-theme`` keeps it; a fragment
    with no ``<html`` tag is returned unchanged.
    """
    if not theme_id:
        return html
    return _HTML_TAG_WITHOUT_THEME_RE.sub(f'<html data-theme="{theme_id}"', html, count=1)


def _resolve_pinned_web_theme() -> str | None:
    """The concrete theme id ``web.theme`` pins, or ``None`` if it names a family.

    Resolved through the design system's shared chain, so the gallery and the
    web terminal cannot disagree about what a configured value means.

    Only a *pin* is stamped. A served artifact page is a follower: it takes a
    server-rendered ``data-theme`` verbatim and has no mode of its own to
    re-resolve, so stamping a family-only value -- which resolves to that
    family's dark id -- would hand a light-OS viewer a dark page, the opposite
    of what an unpinned terminal does with the same config.

    Fails open: an unreadable config or registry must never block the gallery
    from starting, and costs only the pin.
    """
    try:
        from osprey.interfaces.design_system.theme_config import resolve_configured_web_theme

        resolved = resolve_configured_web_theme()
    except FileNotFoundError:
        # No config primed (standalone gallery, tests) — not a fault.
        logger.debug("No config available for web.theme; served pages are unpinned")
        return None
    except Exception:  # config/registry trouble must not block startup
        logger.warning(
            "Could not resolve web.theme for served artifact pages; "
            "they will follow the viewer's own preference",
            exc_info=True,
        )
        return None
    return resolved.id if resolved.pinned_mode else None


def _agent_artifact_dir(artifact_dir: Path, repo_root: Path, base_dir: str) -> str:
    """The spelling of the store's artifact directory handed to the agent.

    Repo-relative exactly when the configured ``agent_data.base_dir`` is
    relative and the directory, taken relative to ``repo_root``, begins with
    that base directory; the absolute directory otherwise. ``~`` counts as
    absolute, because the agent-data root expands it.

    ``artifact_dir.relative_to(repo_root)`` alone is not the test: when the
    base directory does not match the root's tail, ``repo_root_for_agent_data``
    falls back to the root's parent, an ancestor of everything under the root,
    so ``relative_to`` succeeds and yields a wrong-but-plausible path such as
    ``agent/artifacts``. An absolute path is always openable; a relative one is
    only honest when it starts with the configured base directory.

    Args:
        artifact_dir: The directory the store writes artifact files to.
        repo_root: The repo root the store anchors its relative pointers at.
        base_dir: The configured ``agent_data.base_dir``.

    Returns:
        A POSIX repo-relative path, or the absolute directory.
    """
    tail = Path(base_dir).parts
    if (
        not Path(base_dir).expanduser().is_absolute()
        and artifact_dir.is_relative_to(repo_root)
        and artifact_dir.relative_to(repo_root).parts[: len(tail)] == tail
    ):
        return artifact_dir.relative_to(repo_root).as_posix()
    return str(artifact_dir)


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


class _SetInSession(BaseModel):
    """The session a focus was set in: at most one of two fields.

    Browsers send ``session_key``, which the hub resolves through its ledger;
    processes in a conversation send ``transcript_id``, a canonical session
    UUID. Neither, or a key that resolves to no transcript, means "set in no
    session". Both, or a malformed ``transcript_id``, is a 422.
    """

    session_key: str | None = None
    transcript_id: str | None = None

    @model_validator(mode="after")
    def _one_session_field(self) -> _SetInSession:
        from osprey.interfaces.web_terminal.session_key import is_posture_key

        if self.session_key is not None and self.transcript_id is not None:
            raise ValueError("give at most one of session_key and transcript_id")
        if self.transcript_id is not None and not is_posture_key(self.transcript_id):
            raise ValueError(f"{self.transcript_id!r} is not a session UUID")
        return self


class FocusRequest(_SetInSession):
    artifact_id: str
    fullscreen: bool = False
    #: Set by a sender acting on its own (the approval hook), not for a person or
    #: an agent that chose the artifact; broadcast so a layout can tell them apart.
    automatic: bool = False


class BulkDeleteRequest(BaseModel):
    # One pick is a screenful of rows; an unbounded list would hold the
    # index lock for as long as the client asks.
    ids: list[str] = Field(min_length=1, max_length=1000)


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
        with self._lock, suppress(ValueError):
            self._queues.remove(q)

    def broadcast(self, data: dict) -> None:
        """Push data to all connected SSE clients (called from sync context)."""
        with self._lock:
            for q in self._queues:
                # Drop if client is too slow
                with suppress(asyncio.QueueFull):
                    q.put_nowait(data)


#: Used when ``artifact_server.max_timeseries_file_mb`` is unset.
DEFAULT_MAX_TIMESERIES_FILE_MB = 200

#: Used when ``artifact_server.page_size`` is unset.
DEFAULT_PAGE_SIZE = 20
#: Largest page one listing request may ask for; a configured page_size above it is capped.
MAX_PAGE_SIZE = 200


#: How many transcripts one batched ``session-list`` may page at once.
MAX_BATCHED_SESSIONS = 5


class SessionListRequest(BaseModel):
    """``POST /api/artifacts/session-list``: exactly one of the three targets.

    ``session`` names a transcript; ``key`` a session key, listed through the
    ledger's current transcript for it; ``sessions`` up to
    :data:`MAX_BATCHED_SESSIONS` transcripts paged together. ``form``,
    ``runs`` and ``search`` narrow any target the way they narrow
    ``GET /api/artifacts``.
    """

    session: str | None = None
    key: str | None = None
    sessions: list[str] | None = Field(None, min_length=1, max_length=MAX_BATCHED_SESSIONS)
    form: str | None = None
    runs: Literal["grouped"] | None = None
    search: str | None = None
    cursor: str | None = None
    limit: int | None = Field(None, ge=1, le=MAX_PAGE_SIZE)

    @model_validator(mode="after")
    def _one_target(self) -> SessionListRequest:
        given = [name for name in ("session", "key", "sessions") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError("give exactly one of session, key or sessions")
        return self


class SessionSummaryRequest(BaseModel):
    """``POST /api/artifacts/session-summary``: the transcripts one sidebar page shows.

    No cap: the client sends one page of its session rows. Every id must be a
    canonical session UUID (a transcript id has a key's form); anything else
    is a 422.
    """

    sessions: list[str]

    @field_validator("sessions")
    @classmethod
    def _uuid_form(cls, sessions: list[str]) -> list[str]:
        from osprey.interfaces.web_terminal.session_key import is_posture_key

        for transcript_id in sessions:
            if not is_posture_key(transcript_id):
                raise ValueError(f"{transcript_id!r} is not a session UUID")
        return sessions


class _LedgerCache:
    """The hub's session ledger as this process last read it.

    The hub writes ``session-ledger.json`` in this identity's state directory
    (:func:`osprey_connectors.posture_store.state_dir`, the rule the hub's
    ``session_meta.ledger_state_dir`` follows); this app only reads it, again
    whenever the file's mtime or size moves, through
    :func:`session_ledger.load_checked` — the reader the agent's MCP processes
    share. An unreadable ledger is reported as unavailable rather than as an
    empty one (which would silently re-own every artifact): every answer then
    says ``ledger: unavailable`` and only stamped artifacts have an owner. A
    missing file is an empty, available ledger (no session has been recorded
    yet).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, int, int] | None = None
        self._value: tuple[Any, bool] = (None, False)

    def read(self) -> tuple[Any, bool]:
        """``(ledger, available)``; the ledger is empty when it is unavailable."""
        from osprey.stores import session_ledger
        from osprey_connectors import posture_store

        try:
            state_dir = posture_store.state_dir()
        except Exception:  # an unresolvable root is "no ledger", not a crash
            logger.warning("The session ledger's location does not resolve", exc_info=True)
            state_dir = None
        path = session_ledger.ledger_path(state_dir)
        if path is None:
            return session_ledger.Ledger(), False
        try:
            stat = path.stat()
        except FileNotFoundError:
            return session_ledger.Ledger(), True
        except OSError:
            logger.warning("The session ledger at %s cannot be read", path, exc_info=True)
            return session_ledger.Ledger(), False
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key == self._key:
                return self._value
        value = session_ledger.load_checked(state_dir)
        with self._lock:
            self._key, self._value = key, value
        return value


@dataclass(frozen=True)
class _ListingView:
    """What one answer reads besides the entries: run groups, the ledger, the registry."""

    groups: dict[str, Any]
    members: dict[str, list[Any]]
    ledger: Any
    ledger_ok: bool
    registry: Any

    @property
    def owner_ledger(self) -> Any:
        """The ledger :func:`owner_of` reads; none while it is unavailable."""
        return self.ledger if self.ledger_ok else None


def _listing_view(entries: Iterable[Any], ledger: Any, ledger_ok: bool) -> _ListingView:
    """Classify every ``exec_id`` group among *entries* once, for one answer."""
    from osprey.stores.artifact_store import classify, registry_or_default

    members: dict[str, list[Any]] = {}
    for entry in entries:
        if entry.exec_id:
            members.setdefault(entry.exec_id, []).append(entry)
    groups = {exec_id: classify(group) for exec_id, group in members.items()}
    return _ListingView(groups, members, ledger, ledger_ok, registry_or_default())


def _group_of(entry: Any, view: _ListingView) -> Any:
    group = view.groups.get(entry.exec_id) if entry.exec_id else None
    return group if group is not None and entry.id in group.roles else None


def _listed_form(entry: Any, view: _ListingView) -> str | None:
    from osprey.stores.artifact_store import listed_form

    return listed_form(entry, _group_of(entry, view), view.registry)


def _payload(entry: Any, view: _ListingView) -> dict[str, Any]:
    """The entry as every browser-bound frame and response carries it.

    ``entry.to_dict()`` is the index format and stays as it is. Beside it ride
    the server's verdicts, so no client re-derives a rule: ``form`` (the listed
    form, ``None`` for a run's preview or notebook) and its ``form_label``;
    ``run_part`` and the group's ``run_status``; ``owner`` (the transcript that
    owns it, else ``None``); ``operator_visible``; ``interactive``; and
    the legacy ``run_record`` verdict. A run's head also
    lists its ``run_parts``.
    """
    from osprey.stores.artifact_forms import form_label, is_interactive
    from osprey.stores.artifact_store import (
        classify,
        is_run_record,
        operator_visible,
        owner_of,
        run_parts,
        run_role,
    )

    group = _group_of(entry, view)
    form = _listed_form(entry, view)
    if group is not None:
        run_part: str | None = group.roles[entry.id]
        run_status = group.run_status
    else:
        role = run_role(entry)
        run_part = None if role == "output" else role
        run_status = classify([entry]).run_status if role == "result" else None
    payload = {
        **entry.to_dict(),
        "run_record": is_run_record(entry),
        "form": form,
        "form_label": form_label(form, view.registry) if form else None,
        "kind": entry.kind,
        "exec_id": entry.exec_id,
        "run_part": run_part,
        "run_status": run_status,
        "interactive": is_interactive(entry),
        "session_id": entry.session_id,
        "owner": owner_of(entry, view.owner_ledger),
        "operator_visible": operator_visible(entry),
    }
    if group is not None and group.head_id == entry.id:
        payload["run_parts"] = run_parts(group, view.members[entry.exec_id], view.registry)
    return payload


def _refine(
    entries: list[Any],
    view: _ListingView,
    *,
    form: str | None = None,
    runs: str | None = None,
    search: str | None = None,
) -> list[Any]:
    """Keep the entries the ``form``, ``runs`` and ``search`` params admit.

    ``form`` keeps the entries listed under it; ``runs="grouped"`` drops a run's
    preview and notebook (they list under their head; outputs stay);
    ``search`` is a case-insensitive substring of the title, file name,
    description, artifact type or listed form's label.
    """
    from osprey.stores.artifact_forms import form_label

    if not (form or runs or search):
        return entries
    needle = (search or "").lower()
    kept = []
    for entry in entries:
        listed = _listed_form(entry, view)
        if form and listed != form:
            continue
        if runs == "grouped" and listed is None:
            continue
        if needle:
            fields = (
                entry.title,
                entry.filename,
                entry.description,
                entry.artifact_type,
                form_label(listed, view.registry) if listed else "",
            )
            if not any(needle in (value or "").lower() for value in fields):
                continue
        kept.append(entry)
    return kept


def _max_timeseries_file_bytes() -> int:
    """Largest timeseries file the chart/table views will render, in bytes.

    The handler reads the WHOLE file into memory to build a chart or a table,
    so this bound is about the gallery host's memory rather than about the
    data: raising it lets a bigger run be plotted in the browser and costs that
    much resident memory on the machine serving the gallery. A facility whose
    archiver exports run larger than the default says so with
    ``artifact_server.max_timeseries_file_mb``; the file itself is always
    downloadable whatever this says.

    Read on every call — never cached at import — so a test and a re-primed
    process see the value their config declares.
    """
    megabytes = DEFAULT_MAX_TIMESERIES_FILE_MB
    try:
        from osprey_connectors.config import get_config_value

        configured = get_config_value(
            "artifact_server.max_timeseries_file_mb", DEFAULT_MAX_TIMESERIES_FILE_MB
        )
    except Exception:
        logger.debug(
            "No config available for artifact_server.max_timeseries_file_mb", exc_info=True
        )
        configured = None
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        megabytes = configured
    elif configured is not None and configured != DEFAULT_MAX_TIMESERIES_FILE_MB:
        logger.warning(
            "artifact_server.max_timeseries_file_mb must be an integer >= 1 (got %r); using %d MB",
            configured,
            DEFAULT_MAX_TIMESERIES_FILE_MB,
        )
    return megabytes * 1024 * 1024


def _page_size() -> int:
    """How many artifacts one listing request answers with.

    This is therefore how many the gallery draws before it fetches more. A
    facility with slow links or very large stores lowers it with
    ``artifact_server.page_size``; one with a handful of artifacts per shift
    may raise it.
    """
    size = DEFAULT_PAGE_SIZE
    try:
        from osprey_connectors.config import get_config_value

        configured = get_config_value("artifact_server.page_size", DEFAULT_PAGE_SIZE)
    except Exception:
        logger.debug("No config available for artifact_server.page_size", exc_info=True)
        configured = None
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        size = configured
        if size > MAX_PAGE_SIZE:
            logger.warning(
                "artifact_server.page_size is capped at %d (got %d)", MAX_PAGE_SIZE, size
            )
            size = MAX_PAGE_SIZE
    elif configured is not None and configured != DEFAULT_PAGE_SIZE:
        logger.warning(
            "artifact_server.page_size must be an integer >= 1 (got %r); using %d",
            configured,
            DEFAULT_PAGE_SIZE,
        )
    return size


def _union_timestamp_axis(series: dict[str, dict]) -> Collection:
    """Every timestamp any channel carries, deduplicated — the table's row axis.

    Shared by both response formats so ``format=chart``'s ``summary.row_count``
    always matches ``format=table``'s ``total_rows``.
    """
    stamps = chain.from_iterable(data.get("timestamps", []) for data in series.values())
    try:
        return set(stamps)
    except TypeError:
        # An unhashable timestamp (a JSON array, say) cannot go in a set;
        # dedupe by equality instead. `stamps` is one-shot, so rebuild it.
        unique: list = []
        for stamp in chain.from_iterable(data.get("timestamps", []) for data in series.values()):
            if stamp not in unique:
                unique.append(stamp)
        return unique


def _raise_duplicate_sample(channel: str, stamp: object) -> None:
    """Reject two samples sharing one (timestamp, channel) cell."""
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

    Matches by ``==`` rather than by hash; duplicates are rejected eagerly in
    the constructor, matching the hashable path.
    """

    __slots__ = ("_pairs",)

    def __init__(self, channel: str, timestamps: list, values: list) -> None:
        # strict=False: a timestamps/values length mismatch is tolerated, as
        # on the fast path.
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

    Unions every channel's own timestamps into one sorted axis and looks up
    each channel's value at each shared timestamp, leaving ``None`` where
    that channel has no sample. Presentation-only; only the requested page's
    rows are materialized. The returned ``columns`` is the very list each row
    was indexed with — a header taken from any other response can disagree
    with these rows.

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
    value_by_channel: dict[str, dict | _EqualityMatchLookup] = {}
    for ch, data in series.items():
        timestamps = data.get("timestamps", [])
        values = data.get("values", [])
        # A dict of n pairs holds fewer than n entries iff a key repeated.
        try:
            by_ts: dict = dict(zip(timestamps, values, strict=False))
        except TypeError:
            # Unhashable timestamps can't key a dict; match by equality instead.
            value_by_channel[ch] = _EqualityMatchLookup(ch, timestamps, values)
            continue
        if len(by_ts) != min(len(timestamps), len(values)):
            # Re-walk the pairs to name the first repeated timestamp.
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
        # Mutually-incomparable timestamp types (e.g. int vs str) can't sort
        # by `<`; a deterministic string order beats a 500.
        index = sorted(all_timestamps, key=str)

    total_rows = len(index)
    page = index[offset : min(offset + limit, total_rows)]
    rows = [[value_by_channel[ch].get(ts) for ch in columns] for ts in page]
    return columns, page, rows, total_rows


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Artifact Gallery FastAPI application.

    Args:
        workspace_root: Agent-data root containing the ``artifacts/`` dir.
            Omitted, it is the deployment's shared agent-data root, the same
            one the store would resolve, so the store, the index watcher and
            the focus file share one directory.
    """
    from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher
    from osprey.stores.artifact_store import (
        ArtifactEntry,
        ArtifactStore,
        artifact_mutation_actor,
        operator_visible,
        owner_of,
        page_of,
        register_artifact_delete_listener,
        register_artifact_listener,
        unregister_artifact_delete_listener,
        unregister_artifact_listener,
    )
    from osprey_connectors.workspace import resolve_shared_data_root

    data_root: Path = workspace_root if workspace_root is not None else resolve_shared_data_root()
    store = ArtifactStore(workspace_root=data_root)

    # Prime config and load the declared extension types (if available).
    #
    # Resolved through the ordinary config rule, not by looking for a
    # `config.yml` INSIDE the agent-data root — no layout has ever written one
    # there, so the `exists()` gate below was always false and the gallery
    # silently never loaded a custom declaration. `OSPREY_CONFIG` is set on every
    # launch path that starts this app, and resolve_config_path falls back to
    # the render zone beneath the cwd otherwise.
    try:
        from osprey_connectors.workspace import resolve_config_path

        config_path = resolve_config_path()
        if config_path.exists():
            from osprey_connectors.config import get_config_builder

            get_config_builder(config_path=str(config_path), set_as_default=True)
            from osprey.stores.type_registry import load_extension_types

            load_extension_types()
    except Exception as exc:  # config may not be available in all contexts
        logger.warning("Config priming / extension type loading failed (non-fatal): %s", exc)

    # Resolved once, after config priming, and stamped onto every served HTML
    # page (see _stamp_data_theme) so a pinned ``web.theme`` reaches a page
    # opened outside the hub.
    web_theme_pin = _resolve_pinned_web_theme()

    # Resolved once, after config priming like the theme pin: the spelling of
    # the artifact directory that the page's artifactPath hands the agent.
    from osprey_connectors.workspace import agent_data_base_dir, load_osprey_config

    artifact_dir_for_agent = _agent_artifact_dir(
        store.artifact_dir, store.repo_root, agent_data_base_dir(load_osprey_config())
    )

    # Resolved once, after config priming like the theme pin, because the
    # priming above is what points the resolver at this deployment's config.
    facility_timezone = get_facility_timezone().key

    # Resolved once, after config priming like the theme pin, because the
    # listing route's own ``limit`` default is built from it when the route is
    # defined.
    page_size = _page_size()

    broadcaster = _SSEBroadcaster()

    ledger_cache = _LedgerCache()

    def _view() -> _ListingView:
        """The whole index's run groups and the ledger, read once for one answer."""
        return _listing_view(store.list_entries(), *ledger_cache.read())

    def _entry_payload(entry: ArtifactEntry) -> dict[str, Any]:
        """One entry's payload (see :func:`_payload`), reading only its own run group."""
        members = store.run_members(entry.exec_id) if entry.exec_id else []
        return _payload(entry, _listing_view(members, *ledger_cache.read()))

    def _page_body(page: Any, view: _ListingView) -> dict[str, Any]:
        return {
            "count": len(page.entries),
            "total": page.total,
            "next_cursor": page.next_cursor,
            "artifacts": [_payload(e, view) for e in page.entries],
        }

    index_watcher = StoreIndexWatcher(
        workspace_root=data_root,
        broadcaster=broadcaster,
        artifact_store=store,
        payload=_entry_payload,
    )

    def _on_artifact_saved(entry: ArtifactEntry) -> None:
        broadcaster.broadcast({"type": "artifact", **_entry_payload(entry)})

    def _on_artifact_deleted(entry: ArtifactEntry) -> None:
        # Every slot holding the id is cleared, the unscoped one and each transcript's.
        if app.state.focused_artifact_id == entry.id:
            app.state.focused_artifact_id = None
        slots = app.state.focus_by_transcript
        for transcript in [t for t, aid in slots.items() if aid == entry.id]:
            del slots[transcript]
        _write_focus_file()
        broadcaster.broadcast({"type": "artifact_deleted", "id": entry.id})

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
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
    # The directory the AGENT ran in — where its Claude transcript lives.
    # Resolved ONCE here rather than read from `Path.cwd()` per request: the
    # in-process chat/web companion happens to be launched with that cwd, but
    # the standalone `osprey artifacts` gallery is a uvicorn factory runnable
    # from anywhere, and there the per-request read found no transcript and
    # returned an empty audit trail — swallowed by the reader's own
    # exception guard, so the /compose response was simply missing its
    # provenance with nothing to say so.
    app.state.agent_project_dir = deployed_render_dir()
    # Focus slots (contract C6): the unscoped slot, set from no session, and one
    # slot per transcript. A focus replaces only its own slot.
    app.state.focused_artifact_id = None  # None = show latest
    app.state.focus_by_transcript = {}
    import itertools

    from osprey.stores.artifact_scope import SessionScope

    #: When each slot was last set (``None`` = the unscoped slot), so an unkeyed
    #: read answers the most recently set one.
    focus_set_order: dict[str | None, int] = {}
    focus_counter = itertools.count(1)

    def _key_scope(session_key: str) -> SessionScope:
        """The web scope *session_key* names through the hub ledger."""
        from osprey.stores.artifact_scope import resolve_scope

        ledger, ledger_ok = ledger_cache.read()
        return resolve_scope(key=session_key, ledger=ledger, ledger_ok=ledger_ok)

    def _set_in(req: _SetInSession) -> str | None:
        """The transcript a focus was set in; ``None`` for no session."""
        if req.transcript_id is not None:
            return req.transcript_id
        if req.session_key:
            return _key_scope(req.session_key).transcript
        return None

    focus_file = data_root / "focus_state.txt"

    _control_chars = re.compile(r"[\x00-\x1f\x7f]+")
    _marker_opener = re.compile(r"\((id|session)=")

    def _focus_title(title: str) -> str:
        """*title* safe for one focus line: control characters and newlines
        collapse to a space, and ``(id=`` / ``(session=`` lose their ``=`` so no
        title can end the line or spell a marker of its own."""
        return _marker_opener.sub(r"(\1 ", _control_chars.sub(" ", title or ""))

    def _write_focus_file() -> None:
        """Write the focus file the CLI hook reads (contract C7).

        One ``artifact:`` line per transcript slot, ending in that transcript's
        ``(session=<id>)`` marker. The unscoped slot writes no line. Always written, so
        an empty state empties the file rather than leaving a stale one.
        """
        lines: list[str] = []
        for transcript, aid in app.state.focus_by_transcript.items():
            entry = store.get_entry(aid) if aid else None
            if entry is not None:
                lines.append(
                    f'  artifact: "{_focus_title(entry.title)}" (id={aid}) (session={transcript})'
                )
        text = "[Gallery Focus]\n" + "\n".join(lines) + "\n" if lines else ""
        try:
            focus_file.parent.mkdir(parents=True, exist_ok=True)
            focus_file.write_text(text)
        except OSError:
            logger.warning("could not write focus file %s", focus_file, exc_info=True)

    # A hub (re)start holds empty slots: clear any marked lines a previous run left.
    _write_focus_file()

    # --- Routes ---

    @app.get("/")
    @not_content
    async def root(request: Request):
        # A pinned web.theme reaches the gallery shell too. Embedded, the hub's
        # ?theme= outranks it in theme-boot.js's ladder, so this only shows up
        # on a first standalone visit.
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "web_theme_pin": web_theme_pin,
                "facility_timezone": facility_timezone,
                "artifact_dir": artifact_dir_for_agent,
            },
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "artifact_count": len(store.list_entries())}

    @app.get("/api/type-registry")
    async def get_type_registry():
        from osprey.stores.type_registry import registry_to_api_dict, viewer_path

        registry = registry_to_api_dict()
        # ``viewer_url`` only flags "has a viewer": the gallery builds the
        # import URL itself, as a literal the hub's proxy can rebase (a JSON
        # body is never rewritten).
        for ext in registry["extension_types"]:
            if viewer_path(ext["key"]) is not None:
                ext["viewer_url"] = f"/viewers/{quote(ext['key'], safe='')}.js"
        return JSONResponse(registry)

    @app.get("/viewers/{key}.js")
    @not_content
    async def get_viewer(key: str):
        """Extension type ``key``'s viewer module: the one file resolved for it at load.

        Installed facility code, not agent content, hence ``not_content``. The
        key is only a registry lookup and never reaches the filesystem; an
        unknown key or a type without a viewer is a 404.
        """
        from osprey.stores.type_registry import viewer_path

        path = viewer_path(key)
        if path is None or not path.is_file():
            return Response(content="Not found", status_code=404)
        return Response(
            content=path.read_bytes(),
            media_type="text/javascript",
            headers={
                "cache-control": "no-cache, no-store, must-revalidate",
                "x-content-type-options": "nosniff",
            },
        )

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
        limit: int = Query(page_size, ge=1, le=MAX_PAGE_SIZE),
        cursor: str | None = Query(None),
        form: str | None = None,
        after: str | None = None,
        before: str | None = None,
        runs: Literal["grouped"] | None = Query(None),
        exec_id: str | None = None,
    ):
        """List one newest-first page of artifacts.

        ``form`` keeps the entries listed under that form (a run's head under
        ``runs``); ``after``/``before`` keep the half-open window ``[after,
        before)`` of aware ISO-8601 times; ``runs=grouped`` drops a run's
        preview and notebook, which list under their head, and keeps its
        outputs; ``search`` matches the title, file name, description, type
        and form label; ``exec_id`` keeps one run's entries (the chat's run
        rows read a run they have not listed this way). ``total`` and
        ``next_cursor`` count and walk the filtered set.

        This is the unscoped listing, the palette's search and the gallery's
        all-sessions view: it takes no session and keeps dispatch and watchdog
        artifacts. One session's listing is ``POST /api/artifacts/session-list``.
        """
        try:
            entries = store.list_entries(
                type_filter=type,
                after=after,
                before=before,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if exec_id:
            entries = [e for e in entries if e.exec_id == exec_id]
        view = _view()
        entries = _refine(entries, view, form=form, runs=runs, search=search)
        try:
            page = page_of(entries, limit, cursor)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="cursor is not one this listing issued"
            ) from exc
        return _page_body(page, view)

    @app.post("/api/artifacts/session-list")
    async def session_list(body: SessionListRequest):
        """One session's visible artifacts, paged like ``GET /api/artifacts``.

        A transcript's page holds the entries
        :func:`~osprey.stores.artifact_scope.owned_by` its web scope, the rule
        every other listing shares. ``{key}`` resolves the key through
        :func:`~osprey.stores.artifact_scope.resolve_scope` and answers the
        resolved ``transcript`` (``null`` for a key the ledger has no row for,
        or while it is unreadable); ``{sessions}`` answers ``{pages: {id:
        page}}``. An artifact no transcript owns lists under none. Every
        answer says whether the ledger could be read.
        """
        from osprey.stores.artifact_scope import owned_by, resolve_scope, scope_for

        view = _view()
        limit = body.limit or page_size
        entries = store.list_entries()

        def page(selected: list[Any]) -> dict[str, Any]:
            selected = _refine(selected, view, form=body.form, runs=body.runs, search=body.search)
            return _page_body(page_of(selected, limit, body.cursor), view)

        def transcript_page(scope: Any) -> dict[str, Any]:
            return page([e for e in entries if owned_by(e, scope)])

        def owned(transcript_id: str | None) -> dict[str, Any]:
            return transcript_page(scope_for(transcript_id, view.owner_ledger, view.ledger_ok))

        status = "ok" if view.ledger_ok else "unavailable"
        try:
            if body.sessions is not None:
                pages = {tid: owned(tid) for tid in dict.fromkeys(body.sessions)}
                return {"pages": pages, "ledger": status}
            if body.key is not None:
                scope = resolve_scope(key=body.key, ledger=view.ledger, ledger_ok=view.ledger_ok)
                return {**transcript_page(scope), "transcript": scope.transcript, "ledger": status}
            answer = owned(body.session)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="cursor is not one this listing issued"
            ) from exc
        return {**answer, "ledger": status}

    @app.post("/api/artifacts/session-summary")
    async def session_counts(body: SessionSummaryRequest):
        """Counts for one page of sessions, and every owned artifact together.

        One pass over the index, under the same rules as ``session-list``:
        visibility is :func:`~osprey.stores.artifact_store.operator_visible`,
        ownership :func:`~osprey.stores.artifact_store.owner_of`, and an entry
        counts under its listed form, so ``by_form[f]`` equals the rows
        ``session-list`` returns with ``runs=grouped`` and ``form=f`` and a run
        counts once (its head) whatever its status. ``by_form`` names only the
        forms that occur. ``owned_total`` counts every owned entry, requested or
        not: the client's "More sessions" is ``owned_total`` minus its loaded
        totals. An entry no transcript owns counts nowhere.
        """
        view = _view()
        ledger = view.owner_ledger
        requested = dict.fromkeys(body.sessions)
        counts: dict[str, dict[str, Any]] = {tid: {"total": 0, "by_form": {}} for tid in requested}
        owned_total = 0
        for entry in store.list_entries():
            if not operator_visible(entry):
                continue
            form = _listed_form(entry, view)
            if form is None:  # a run's preview or notebook lists under its head
                continue
            owner = owner_of(entry, ledger)
            if owner is None:
                continue
            owned_total += 1
            bucket = counts.get(owner)
            if bucket is None:
                continue
            bucket["total"] += 1
            bucket["by_form"][form] = bucket["by_form"].get(form, 0) + 1
        return {
            "counts": counts,
            "owned_total": owned_total,
            "ledger": "ok" if view.ledger_ok else "unavailable",
        }

    @app.get("/api/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        entry = store.get_entry(artifact_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        return _entry_payload(entry)

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
            # A store on disk holds every shape any OSPREY release ever wrote,
            # so data_file may be (a) a repo-root-relative path like
            # "var/agent_data/artifacts/foo.json" (the ArtifactStore format),
            # (b) a bare filename, or (c) some other workspace-relative path.
            # Try each candidate; the absolute strings a DataContext-era entry
            # carries are handled by the is_absolute() branch above.
            candidates = [
                store.repo_root / filepath,
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
        if file_size > _max_timeseries_file_bytes():
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
            # Cross-channel totals are computed server-side: per-channel point
            # sums disagree with the unioned row axis, and `row_count` is not
            # derivable client-side at all.
            return {
                "channels": channels_out,
                "metadata": query_meta,
                "summary": {
                    "total_points": sum(ch["total_points"] for ch in channels_out),
                    "returned_points": sum(ch["returned_points"] for ch in channels_out),
                    "downsampled": any(
                        ch["returned_points"] < ch["total_points"] for ch in channels_out
                    ),
                    "row_count": len(_union_timestamp_axis(series)),
                },
            }

        # format == "table": pivot per-channel series onto a unioned row axis.
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
    async def get_focus(session_key: str | None = None):
        """The focus a view shows (contract C6).

        With ``session_key``: that key's current transcript's slot, else the
        newest entry that key's scope owns, else none. Without it: the most
        recently set slot of any kind, else the newest operator-visible entry.
        A slot whose artifact is gone is cleared on the way.
        """
        from osprey.stores.artifact_scope import owned_by
        from osprey.stores.artifact_store import operator_visible

        def _slot_entry(slot: str | None) -> ArtifactEntry | None:
            if slot is None:
                focused_id = app.state.focused_artifact_id
            else:
                focused_id = app.state.focus_by_transcript.get(slot)
            if not focused_id:
                return None
            entry = store.get_entry(focused_id)
            if entry is None:  # stale: the artifact went away under the slot
                if slot is None:
                    app.state.focused_artifact_id = None
                else:
                    app.state.focus_by_transcript.pop(slot, None)
                focus_set_order.pop(slot, None)
            return entry

        if session_key is not None:
            scope = _key_scope(session_key)
            if scope.transcript is None:
                return {"focused": False, "artifact": None}
            entry = _slot_entry(scope.transcript)
            if entry is not None:
                return {"focused": True, "artifact": _entry_payload(entry)}
            owned = [e for e in store.list_entries() if owned_by(e, scope)]
            return {"focused": False, "artifact": _entry_payload(owned[-1]) if owned else None}

        slots: list[str | None] = [None, *app.state.focus_by_transcript]
        slots.sort(key=lambda slot: focus_set_order.get(slot, 0), reverse=True)
        for slot in slots:
            entry = _slot_entry(slot)
            if entry is not None:
                return {"focused": True, "artifact": _entry_payload(entry)}

        visible = [e for e in store.list_entries() if operator_visible(e)]
        return {"focused": False, "artifact": _entry_payload(visible[-1]) if visible else None}

    @app.post("/api/focus")
    async def set_focus(req: FocusRequest):
        entry = store.get_entry(req.artifact_id)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {req.artifact_id} not found",
            )
        transcript = _set_in(req)
        if transcript is None:
            app.state.focused_artifact_id = req.artifact_id
        else:
            app.state.focus_by_transcript[transcript] = req.artifact_id
        focus_set_order[transcript] = next(focus_counter)
        _write_focus_file()
        event: dict[str, Any] = {
            "type": "focus",
            "domain": "artifact",
            "id": req.artifact_id,
            "session": transcript,
        }
        if req.fullscreen:
            event["fullscreen"] = True
        if req.automatic:
            event["automatic"] = True
        broadcaster.broadcast(event)
        return {"status": "ok", "artifact_id": req.artifact_id}

    @app.get("/files/{artifact_id}/{filename}")
    async def serve_file(
        artifact_id: str,
        filename: str,  # noqa: ARG001 - route path parameter; the download name comes from the artifact entry
    ):
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
                headers=document_headers(media_type),
            )

        # HTML types get the responsive/theming snippet and a self-contained
        # Plotly. The page is served into the content sandbox's opaque origin,
        # where every server path answers 401 on a multi-user deployment, so
        # nothing it loads may be a /static or /design-system URL.
        offline = is_offline()
        content = filepath.read_bytes()
        # Artifacts may have been generated with include_plotlyjs='cdn' (or
        # pointing at a local /static copy) whatever OSPREY's own code paths
        # use: offline the CDN is unreachable and a /static path is never
        # loadable in the sandbox, so those tags become one inline copy.
        html = inline_assets.self_contain_plotly(
            content.decode("utf-8", errors="replace"), offline=offline
        )
        content = html.encode("utf-8")
        snippet = inline_assets.self_contained(snippet)
        # Supply the Plotly bundle unless the page already loads one, by
        # script tag or inlined (include_plotlyjs=True). Avoid duplicates —
        # the bundle is 4.8 MB.
        if entry.artifact_type == "plot_html" and not html_has_plotly_bundle(content):
            snippet = inline_assets.plotly_script(offline=offline) + "\n" + snippet
        content = _inject_html_snippet(content, snippet)
        page = _stamp_data_theme(content.decode("utf-8", errors="replace"), web_theme_pin)
        return Response(
            content=page.encode("utf-8"),
            media_type=entry.mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{entry.filename}"',
                **document_headers(entry.mime_type),
            },
        )

    @app.delete("/api/artifacts/{artifact_id}")
    async def delete_artifact(artifact_id: str):
        # This delete is a person clicking in the gallery, not the agent —
        # tag it so store listeners don't report it as agent activity.
        # Deleting a run's head deletes its whole run; anything else goes alone.
        with artifact_mutation_actor("human"):
            removed = store.delete_with_group(artifact_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        return {"status": "ok", "artifact_id": artifact_id}

    @app.post("/api/artifacts/delete")
    async def delete_artifacts(body: BulkDeleteRequest):
        # This delete is a person clicking in the gallery, not the agent —
        # tag it so store listeners don't report it as agent activity.
        with artifact_mutation_actor("human"):
            removed = store.delete_entries(body.ids)
        removed_ids = {e.id for e in removed}
        missing = [i for i in dict.fromkeys(body.ids) if i not in removed_ids]
        return {"deleted": [e.id for e in removed], "missing": missing}

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
            html_bytes = _inject_html_snippet(html.encode("utf-8"), _NOTEBOOK_SNIPPET)
            return HTMLResponse(
                content=_stamp_data_theme(html_bytes.decode("utf-8"), web_theme_pin),
                headers=opaque_headers(),
            )
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
        return HTMLResponse(
            content=_stamp_data_theme(html, web_theme_pin), headers=opaque_headers()
        )

    # Logbook entry composer
    from osprey.interfaces.artifacts.logbook import logbook_router

    app.include_router(logbook_router)

    # Added before the shared stack, so it sits inside the auth gate (which
    # must stay outermost) and against the router: every response the app
    # produces that is not marked ``not_content`` -- the content routes, any
    # route added later, error statuses and unmatched paths alike -- is
    # served into an opaque sandbox origin.
    app.add_middleware(ContentSandboxMiddleware)
    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
