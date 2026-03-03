"""OSPREY Artifact Gallery — FastAPI Application.

A unified gallery for interactive artifacts (plots, tables, HTML, markdown)
produced by Claude during analysis sessions.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from osprey.utils.timeseries import extract_timeseries_frame, lttb_downsample

STATIC_DIR = Path(__file__).parent / "static"

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


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Artifact Gallery FastAPI application.

    Args:
        workspace_root: Workspace root containing ``artifacts/`` dir.
            Defaults to ``./osprey-workspace``.
    """
    from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher
    from osprey.stores.artifact_store import (
        ArtifactEntry,
        ArtifactStore,
        register_artifact_listener,
        unregister_artifact_listener,
    )

    store = ArtifactStore(workspace_root=workspace_root)
    broadcaster = _SSEBroadcaster()

    index_watcher = StoreIndexWatcher(
        workspace_root=workspace_root,
        broadcaster=broadcaster,
        artifact_store=store,
    )

    def _on_artifact_saved(entry: ArtifactEntry) -> None:
        broadcaster.broadcast({"type": "artifact", **entry.to_dict()})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        register_artifact_listener(_on_artifact_saved)
        index_watcher.start()
        yield
        index_watcher.stop()
        unregister_artifact_listener(_on_artifact_saved)

    app = FastAPI(
        title="OSPREY Artifact Gallery",
        description="Interactive gallery for analysis artifacts",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

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
            # New unified artifacts store files in the artifacts/ subdirectory;
            # legacy DataContext entries used absolute paths.
            candidate = store._store_dir / filepath
            if candidate.exists():
                filepath = candidate
            else:
                filepath = store._workspace / filepath
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
        frame, query_meta = extract_timeseries_frame(raw)
        columns = frame.get("columns", [])
        index = frame.get("index", [])
        rows = frame.get("data", [])
        total_rows = len(index)

        if format == "chart":
            ds_index, ds_rows = lttb_downsample(index, rows, max_points)
            return {
                "columns": columns,
                "index": ds_index,
                "data": ds_rows,
                "total_rows": total_rows,
                "downsampled": len(ds_index) < total_rows,
                "returned_points": len(ds_index),
                "metadata": query_meta,
            }

        # format == "table"
        end = min(offset + limit, total_rows)
        sliced_index = index[offset:end]
        sliced_data = rows[offset:end]
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
            return FileResponse(
                filepath,
                media_type=entry.mime_type,
                filename=entry.filename,
                content_disposition_type="inline",
            )

        # HTML types may need responsive snippet injection
        content = filepath.read_bytes()
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
        if app.state.focused_artifact_id == artifact_id:
            app.state.focused_artifact_id = None
            _write_focus_file()
        broadcaster.broadcast({"type": "artifact_deleted", "id": artifact_id})
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

    from osprey.interfaces.common_middleware import NoCacheStaticMiddleware

    app.add_middleware(NoCacheStaticMiddleware)

    # Mount static assets
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
