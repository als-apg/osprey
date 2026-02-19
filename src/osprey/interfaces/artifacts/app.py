"""OSPREY Artifact Gallery — FastAPI Application.

A lightweight web gallery that serves interactive artifacts (plots, tables,
HTML, markdown) produced by Claude during analysis sessions, a browsable
index of all MCP tool output data (the "data context"), and a memory domain
for persistent session memories (notes and pins).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"

# Snippet injected into Plotly/table/generic HTML artifacts so they fill the
# iframe viewport in Focus Mode.  CSS alone is not enough for Plotly because
# Plotly.newPlot() applies layout.width/height via JS *after* load, overriding
# CSS.  We therefore inject a script that deletes those fixed dimensions and
# calls Plotly.Plots.resize() once the library is ready.
_RESPONSIVE_PLOTLY = """<style>
/* OSPREY: fill iframe viewport */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
.plotly-graph-div { width: 100% !important; height: 100vh !important; }
.js-plotly-plot { width: 100% !important; height: 100vh !important; }
table { max-width: 100%; }
</style>
<script>
/* OSPREY: remove Plotly's hardcoded pixel dimensions so responsive mode works */
(function(){
  function resizeAll() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
      if (gd.layout) {
        delete gd.layout.width;
        delete gd.layout.height;
      }
      if (typeof Plotly !== 'undefined') {
        Plotly.Plots.resize(gd);
      }
    });
  }
  /* Plotly CDN may still be loading; wait for it, then resize. */
  if (document.readyState === 'complete') { setTimeout(resizeAll, 50); }
  else { window.addEventListener('load', function(){ setTimeout(resizeAll, 50); }); }
  window.addEventListener('resize', resizeAll);
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
/* OSPREY: make notebook fill iframe viewport */
html, body { margin: 0; padding: 0; width: 100%; height: 100%; }
body.jp-Notebook { padding: 0 16px; overflow: auto; }
.jp-Cell { max-width: 100%; }
/* Classic nbconvert fallback */
#notebook-container, .container { max-width: 100% !important; width: 100% !important; padding: 0 16px; }
</style>"""

_RESPONSIVE_SNIPPETS = {
    "plot_html": _RESPONSIVE_PLOTLY,
    "table_html": _RESPONSIVE_TABLE_HTML,
    "html": _RESPONSIVE_TABLE_HTML,
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


class ContextFocusRequest(BaseModel):
    entry_id: int


class MemoryFocusRequest(BaseModel):
    memory_id: int


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    tags: list[str] | None = None
    importance: str | None = None


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


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Artifact Gallery FastAPI application.

    Args:
        workspace_root: Workspace root containing ``artifacts/`` dir.
            Defaults to ``./osprey-workspace``.
    """
    from osprey.mcp_server.artifact_store import (
        ArtifactEntry,
        ArtifactStore,
        register_artifact_listener,
        unregister_artifact_listener,
    )
    from osprey.mcp_server.data_context import (
        DataContext,
        DataContextEntry,
        register_context_listener,
        unregister_context_listener,
    )
    from osprey.mcp_server.memory_store import (
        MemoryEntry,
        MemoryStore,
        register_memory_listener,
        unregister_memory_listener,
    )

    from osprey.interfaces.artifacts.store_watcher import StoreIndexWatcher

    store = ArtifactStore(workspace_root=workspace_root)
    context_store = DataContext(workspace_root=workspace_root)
    memory_store = MemoryStore(workspace_root=workspace_root)
    broadcaster = _SSEBroadcaster()

    index_watcher = StoreIndexWatcher(
        workspace_root=workspace_root,
        broadcaster=broadcaster,
        context_store=context_store,
        artifact_store=store,
        memory_store=memory_store,
    )

    def _on_artifact_saved(entry: ArtifactEntry) -> None:
        broadcaster.broadcast({"type": "artifact", **entry.to_dict()})

    def _on_context_saved(entry: DataContextEntry) -> None:
        broadcaster.broadcast({"type": "context", **entry.to_dict()})

    def _on_memory_saved(entry: MemoryEntry) -> None:
        broadcaster.broadcast({"type": "memory", **entry.to_dict()})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        register_artifact_listener(_on_artifact_saved)
        register_context_listener(_on_context_saved)
        register_memory_listener(_on_memory_saved)
        index_watcher.start()
        yield
        index_watcher.stop()
        unregister_artifact_listener(_on_artifact_saved)
        unregister_context_listener(_on_context_saved)
        unregister_memory_listener(_on_memory_saved)

    app = FastAPI(
        title="OSPREY Artifact Gallery",
        description="Interactive gallery for analysis artifacts",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references for route handlers
    app.state.artifact_store = store
    app.state.focused_artifact_id = None  # None = show latest
    app.state.context_store = context_store
    app.state.focused_context_id = None
    app.state.memory_store = memory_store
    app.state.focused_memory_id = None

    focus_file = workspace_root / "focus_state.txt"

    def _write_focus_file() -> None:
        """Write current focus state to a plain-text file for the CLI hook."""
        lines: list[str] = []
        aid = app.state.focused_artifact_id
        if aid:
            entry = store.get_entry(aid)
            if entry:
                lines.append(f'  artifact: "{entry.title}" (id={aid})')
        cid = app.state.focused_context_id
        if cid is not None:
            entry = context_store.get_entry(cid)
            if entry:
                lines.append(f"  context:  #{cid} {entry.description[:80]}")
        mid = app.state.focused_memory_id
        if mid is not None:
            entry = memory_store.get_entry(mid)
            if entry:
                lines.append(f"  memory:   #{mid} {entry.content[:60]}")
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
    async def list_artifacts(type: str | None = None, search: str | None = None):
        entries = store.list_entries(type_filter=type, search=search)
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
        broadcaster.broadcast({"type": "focus", "domain": "artifact", "id": req.artifact_id})
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
            from osprey.mcp_server.notebook_renderer import get_or_render_html

            cache_dir = store.artifact_dir / "_notebook_cache"
            html, _ = get_or_render_html(filepath, cache_dir=cache_dir)
            html_bytes = _inject_html_snippet(
                html.encode("utf-8"), _NOTEBOOK_RESPONSIVE_CSS
            )
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

        # Always use read container (port 8088) for safety.
        # Use /doc/tree/ (single-document mode) instead of /lab/tree/ so the
        # notebook opens directly without the file browser sidebar and
        # immediately triggers kernel selection.
        jupyter_path = f"artifacts/{entry.filename}"
        jupyter_url = f"http://127.0.0.1:8088/doc/tree/{jupyter_path}"

        return {
            "jupyter_url": jupyter_url,
            "artifact_id": artifact_id,
        }

    # --- Context routes ---
    # NOTE: /api/context/focus must be registered BEFORE /api/context/{entry_id}
    # so FastAPI doesn't try to parse "focus" as an int path parameter.

    @app.get("/api/context")
    async def list_context(
        tool: str | None = None,
        data_type: str | None = None,
        search: str | None = None,
    ):
        entries = context_store.list_entries(
            tool_filter=tool,
            data_type_filter=data_type,
            search=search,
        )
        return {
            "count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }

    @app.get("/api/context/focus")
    async def get_context_focus():
        focused_id = app.state.focused_context_id
        if focused_id is not None:
            entry = context_store.get_entry(focused_id)
            if entry:
                return {"focused": True, "entry": entry.to_dict()}
            # Stale focus — clear it and fall back to latest
            app.state.focused_context_id = None

        # Fall back to latest context entry
        entries = context_store.list_entries()
        if entries:
            return {"focused": False, "entry": entries[-1].to_dict()}
        return {"focused": False, "entry": None}

    @app.post("/api/context/focus")
    async def set_context_focus(req: ContextFocusRequest):
        entry = context_store.get_entry(req.entry_id)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Context entry {req.entry_id} not found",
            )
        app.state.focused_context_id = req.entry_id
        _write_focus_file()
        broadcaster.broadcast({"type": "focus", "domain": "context", "id": req.entry_id})
        return {"status": "ok", "entry_id": req.entry_id}

    @app.get("/api/context/{entry_id}")
    async def get_context_entry(entry_id: int):
        entry = context_store.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Context entry {entry_id} not found")
        return entry.to_dict()

    @app.get("/api/context/{entry_id}/data")
    async def get_context_data(entry_id: int):
        filepath = context_store.get_file_path(entry_id)
        if not filepath:
            raise HTTPException(
                status_code=404,
                detail=f"Context entry {entry_id} not found or data file missing",
            )
        return Response(
            content=filepath.read_bytes(),
            media_type="application/json",
        )

    @app.get("/api/context/{entry_id}/image")
    async def get_context_image(entry_id: int):
        """Serve the referenced image file from a context entry.

        For entries like screenshots where the JSON data file contains a
        ``data.filepath`` pointing to an image on disk, this endpoint reads
        that path and serves the image with the correct MIME type.
        """
        filepath = context_store.get_file_path(entry_id)
        if not filepath:
            raise HTTPException(
                status_code=404,
                detail=f"Context entry {entry_id} not found or data file missing",
            )
        try:
            payload = json.loads(filepath.read_text())
            image_path = Path(payload.get("data", {}).get("filepath", ""))
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=400, detail="Cannot parse data file")

        if not image_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Referenced image not found: {image_path}"
            )

        suffix = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(image_path, media_type=media_type)

    @app.delete("/api/context/{entry_id}")
    async def delete_context_entry(entry_id: int):
        deleted = context_store.delete_entry(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Context entry {entry_id} not found")
        if app.state.focused_context_id == entry_id:
            app.state.focused_context_id = None
            _write_focus_file()
        broadcaster.broadcast({"type": "context_deleted", "id": entry_id})
        return {"status": "ok", "entry_id": entry_id}

    # --- Memory routes ---
    # NOTE: /api/memory/focus must be registered BEFORE /api/memory/{memory_id}
    # so FastAPI doesn't try to parse "focus" as an int path parameter.

    @app.get("/api/memory")
    async def list_memories(
        type: str | None = None,
        tag: str | None = None,
        importance: str | None = None,
        search: str | None = None,
    ):
        tags = [tag] if tag else None
        entries = memory_store.list_entries(
            memory_type=type,
            tags=tags,
            importance=importance,
            search=search,
        )
        return {
            "count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }

    @app.get("/api/memory/focus")
    async def get_memory_focus():
        focused_id = app.state.focused_memory_id
        if focused_id is not None:
            entry = memory_store.get_entry(focused_id)
            if entry:
                return {"focused": True, "entry": entry.to_dict()}
            # Stale focus — clear it and fall back to latest
            app.state.focused_memory_id = None

        # Fall back to latest memory entry
        entries = memory_store.list_entries()
        if entries:
            return {"focused": False, "entry": entries[-1].to_dict()}
        return {"focused": False, "entry": None}

    @app.post("/api/memory/focus")
    async def set_memory_focus(req: MemoryFocusRequest):
        entry = memory_store.get_entry(req.memory_id)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"Memory {req.memory_id} not found",
            )
        app.state.focused_memory_id = req.memory_id
        _write_focus_file()
        broadcaster.broadcast({"type": "focus", "domain": "memory", "id": req.memory_id})
        return {"status": "ok", "memory_id": req.memory_id}

    @app.get("/api/memory/{memory_id}")
    async def get_memory(memory_id: int):
        entry = memory_store.get_entry(memory_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        return entry.to_dict()

    @app.patch("/api/memory/{memory_id}")
    async def update_memory(memory_id: int, req: MemoryUpdateRequest):
        fields: dict[str, Any] = {}
        if req.content is not None:
            fields["content"] = req.content
        if req.tags is not None:
            fields["tags"] = req.tags
        if req.importance is not None:
            fields["importance"] = req.importance

        if not fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        entry = memory_store.update_entry(memory_id, **fields)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        return entry.to_dict()

    @app.delete("/api/memory/{memory_id}")
    async def delete_memory(memory_id: int):
        deleted = memory_store.delete_entry(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        if app.state.focused_memory_id == memory_id:
            app.state.focused_memory_id = None
            _write_focus_file()
        broadcaster.broadcast({"type": "memory_deleted", "id": memory_id})
        return {"status": "ok", "memory_id": memory_id}

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
