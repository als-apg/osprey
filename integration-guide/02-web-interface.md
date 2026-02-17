# Recipe 2: Adding a Web Interface

## When You Need This

You want a browser-based UI for interactive analysis, visualization, or data management. In OSPREY, every web interface is a self-contained FastAPI application with vanilla JavaScript frontend.

## The Pattern

```
src/osprey/interfaces/{name}/
├── __init__.py          # Public exports: create_app, run_server
├── app.py               # FastAPI app factory + lifespan + run_server()
├── api/
│   ├── __init__.py
│   ├── routes.py        # HTTP route handlers (REST endpoints)
│   └── schemas.py       # Pydantic request/response models
├── static/
│   ├── index.html       # Single-page app shell
│   ├── css/
│   │   ├── variables.css   # Design tokens
│   │   ├── base.css        # Reset, typography, utilities
│   │   ├── layout.css      # Header, main, grid
│   │   └── components.css  # Buttons, cards, inputs, badges
│   └── js/
│       ├── app.js          # Main coordinator, routing, init
│       ├── api.js          # HTTP client layer
│       └── {feature}.js    # One module per feature area
└── mcp/                 # Co-located MCP server (optional, see Recipe 1)
```

## File-by-File Breakdown

### `app.py` — App Factory + Lifespan

```python
"""Web interface for {your domain}."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _create_lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # --- Startup ---
    logger.info("Starting {name} web interface")

    # Initialize your service (store in app.state for route access)
    from .service import create_my_service
    app.state.my_service = await create_my_service(app.state.config)

    yield

    # --- Shutdown ---
    logger.info("Shutting down {name} web interface")
    if hasattr(app.state, "my_service"):
        await app.state.my_service.close()


def create_app(config_path: str | None = None) -> FastAPI:
    """App factory — returns a configured FastAPI instance."""
    app = FastAPI(
        title="{Name} Interface",
        lifespan=_create_lifespan,
    )

    # CORS (permissive for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load config and store on app.state
    from osprey.mcp_server.common import load_osprey_config
    raw = load_osprey_config(config_path)
    app.state.config = raw.get("my_section", {})

    # Register API routes
    from .api.routes import router
    app.include_router(router, prefix="/api")

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "{name}"}

    # Serve index.html at root
    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    # Mount static files LAST (catch-all)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def run_server(host: str = "127.0.0.1", port: int = 8088) -> None:
    """Direct entry point for CLI commands."""
    uvicorn.run(create_app(), host=host, port=port)
```

**Key points:**
- **App factory pattern** — `create_app()` returns a configured app, never modifies global state
- **Lifespan context manager** — handles startup/shutdown lifecycle, not `@app.on_event`
- **`app.state`** — the dependency injection mechanism. Routes access services via `request.app.state.my_service`
- **Static files mounted last** — prevents the catch-all from shadowing API routes
- **Health endpoint** — always present, used by auto-launch and monitoring

### `api/routes.py` — HTTP Route Handlers

```python
"""API routes for {your domain}."""

from fastapi import APIRouter, HTTPException, Request

from .schemas import MyRequest, MyResponse

router = APIRouter()


@router.get("/status")
async def get_status(request: Request):
    """Service health and configuration."""
    service = request.app.state.my_service
    status = await service.get_status()
    return status


@router.post("/analyze", response_model=MyResponse)
async def analyze(request: Request, body: MyRequest):
    """Run analysis."""
    service = request.app.state.my_service
    try:
        result = await service.analyze(body.query, body.parameters)
        return MyResponse.from_result(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.get("/items")
async def list_items(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "timestamp",
):
    """Paginated item listing."""
    service = request.app.state.my_service
    offset = (page - 1) * page_size
    items = await service.list_items(offset=offset, limit=page_size, sort_by=sort_by)
    total = await service.count_items()
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
    }
```

**API conventions:**
- Service access via `request.app.state.{service_name}`
- Errors via `HTTPException` with `detail` message
- Pagination: `page` + `page_size` query params, response includes `total`
- Date params: ISO 8601 strings, parsed in service layer
- File serving: Use `FileResponse` with correct `media_type`

### `api/schemas.py` — Pydantic Models

```python
"""Request/response schemas for {your domain}."""

from enum import Enum

from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    FFT = "fft"
    NAFF = "naff"
    SVD = "svd"


class MyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Analysis query")
    mode: AnalysisMode = Field(default=AnalysisMode.FFT, description="Analysis method")
    max_results: int = Field(default=10, ge=1, le=100)
    parameters: dict = Field(default_factory=dict)


class MyResponse(BaseModel):
    items: list[dict]
    total: int
    mode_used: str
    execution_time_ms: float | None = None
```

**Schema conventions:**
- Enums for fixed option sets (modes, statuses)
- `Field(...)` for required, `Field(default=...)` for optional
- Validation via Pydantic: `ge`, `le`, `min_length`, `max_length`
- Response models mirror what the frontend expects

## `__init__.py` — Public Exports

```python
"""Web interface for {your domain}."""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
```

Always export `create_app` and `run_server` — the CLI command uses these.

## Serving Binary Files

If your interface serves binary data (images, HDF5 files, attachments):

```python
from fastapi.responses import Response

@router.get("/files/{file_id}")
async def serve_file(request: Request, file_id: str):
    service = request.app.state.my_service
    file_data = await service.get_file(file_id)
    if file_data is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    return Response(
        content=file_data["data"],
        media_type=file_data["mime_type"],
        headers={"Content-Disposition": f'inline; filename="{file_data["filename"]}"'},
    )
```

## Path Traversal Protection

Any endpoint that serves files from disk MUST validate paths:

```python
from pathlib import Path

ALLOWED_ROOT = Path("/path/to/workspace")

@router.get("/files/content/{filepath:path}")
async def get_file_content(filepath: str):
    target = (ALLOWED_ROOT / filepath).resolve()
    if not target.is_relative_to(ALLOWED_ROOT):
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)
```

## Auto-Launch Pattern

If your interface should start automatically (like the Artifact Gallery does on first artifact creation):

```python
"""Auto-launcher for {name} interface."""

import threading
import time
import logging
import requests

logger = logging.getLogger(__name__)

_launch_lock = threading.Lock()
_launched = False


def ensure_server_running(host: str = "127.0.0.1", port: int = 8088) -> None:
    """Start the server if not already running."""
    global _launched

    with _launch_lock:
        if _launched:
            return

        # Check if already running
        try:
            resp = requests.get(f"http://{host}:{port}/health", timeout=2)
            if resp.status_code == 200:
                _launched = True
                return
        except requests.ConnectionError:
            pass

        # Launch in daemon thread
        from .app import run_server
        thread = threading.Thread(
            target=run_server,
            kwargs={"host": host, "port": port},
            daemon=True,
        )
        thread.start()
        _launched = True

        # Wait for readiness
        for _ in range(30):
            try:
                resp = requests.get(f"http://{host}:{port}/health", timeout=1)
                if resp.status_code == 200:
                    logger.info(f"{{name}} server ready at http://{host}:{port}")
                    return
            except requests.ConnectionError:
                time.sleep(0.5)
```

## SSE (Server-Sent Events) for Real-Time Updates

If your interface needs real-time updates (new data arriving, analysis progress):

```python
import asyncio
from fastapi.responses import StreamingResponse


# Event broadcaster (module-level)
_listeners: list[asyncio.Queue] = []


def broadcast_event(event_type: str, data: dict) -> None:
    """Send event to all connected SSE clients."""
    import json
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for queue in _listeners:
        queue.put_nowait(message)


@router.get("/events")
async def event_stream():
    """SSE endpoint for real-time updates."""
    queue: asyncio.Queue = asyncio.Queue()
    _listeners.append(queue)

    async def generate():
        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            _listeners.remove(queue)

    return StreamingResponse(generate(), media_type="text/event-stream")
```

Frontend consumption:

```javascript
const eventSource = new EventSource('/api/events');
eventSource.addEventListener('new_data', (event) => {
    const data = JSON.parse(event.data);
    updateUI(data);
});
```

## Concrete Reference

**ARIEL web interface** (`src/osprey/interfaces/ariel/`):
- `app.py` — 150 lines, full app factory with lifespan, config loading, health check
- `api/routes.py` — 350 lines, search, entries, filter-options, config management, file serving
- `api/schemas.py` — Pydantic models for all request/response types
- `api/drafts.py` — File-based draft storage with TTL cleanup
- `static/` — Full SPA with 10 JS modules, 6 CSS files

**Artifact Gallery** (`src/osprey/interfaces/artifacts/`):
- `app.py` — SSE event broadcasting, responsive HTML injection for Plotly/tables
- Auto-launch from MCP tool on first artifact creation

## Checklist

- [ ] `app.py` with `create_app()` factory and `_create_lifespan()` context manager
- [ ] `__init__.py` exporting `create_app` and `run_server`
- [ ] `api/routes.py` with FastAPI router, service access via `request.app.state`
- [ ] `api/schemas.py` with Pydantic request/response models
- [ ] `/health` endpoint always present
- [ ] `/` serves `index.html` via `FileResponse`
- [ ] Static files mounted with `StaticFiles` (LAST, after API routes)
- [ ] Path traversal protection on any file-serving endpoints
- [ ] CORS middleware configured
- [ ] SSE endpoint if real-time updates needed
