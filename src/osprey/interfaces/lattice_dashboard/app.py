"""Lattice Dashboard — FastAPI application.

Serves the dashboard SPA, REST API for lattice state management,
and SSE stream for live figure updates.

Usage::

    from osprey.interfaces.lattice_dashboard.app import create_app
    app = create_app(workspace_root=Path("./osprey-workspace"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import ALL_FIGURES, LatticeState

logger = logging.getLogger("osprey.lattice_dashboard")

STATIC_DIR = Path(__file__).parent / "static"


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
        with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(data)
                except asyncio.QueueFull:
                    pass


# ── Request models ────────────────────────────────────────


class InitRequest(BaseModel):
    lattice_path: str


class ParamRequest(BaseModel):
    family: str
    value: float


# ── App factory ───────────────────────────────────────────


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Lattice Dashboard FastAPI application.

    Args:
        workspace_root: Workspace root (e.g. ``./osprey-workspace``).
            The lattice state lives under ``<workspace>/lattice/``.
    """
    ws_root = Path(workspace_root) if workspace_root else Path("./osprey-workspace")
    state_dir = ws_root / "lattice"

    state = LatticeState(state_dir)
    broadcaster = _SSEBroadcaster()
    compute = ComputeManager(state, broadcaster)

    app = FastAPI(
        title="Lattice Dashboard",
        description="Live lattice visualization dashboard",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "lattice_dashboard"}

    # ── State API ─────────────────────────────────────────

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return state.load()

    @app.post("/api/state/init")
    async def init_lattice(body: InitRequest) -> dict[str, Any]:
        try:
            result = state.initialize(body.lattice_path)
        except Exception as exc:
            logger.exception("Failed to initialize lattice")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        broadcaster.broadcast(
            {
                "type": "state_updated",
                "summary": result.get("summary", {}),
                "families": result.get("families", {}),
            }
        )

        # Auto-refresh fast figures after init
        compute.refresh_fast()

        return result

    @app.post("/api/state/param")
    async def set_param(body: ParamRequest) -> dict[str, Any]:
        current = state.load()
        if body.family not in current.get("families", {}):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown family: {body.family}",
            )

        result = state.set_param(body.family, body.value)
        broadcaster.broadcast(
            {
                "type": "state_updated",
                "summary": result.get("summary", {}),
                "families": result.get("families", {}),
            }
        )
        return result

    # ── Refresh API ───────────────────────────────────────

    @app.post("/api/refresh")
    async def refresh_fast() -> dict[str, Any]:
        launched = compute.refresh_fast()
        return {"status": "ok", "launched": launched}

    @app.post("/api/refresh/{figure}")
    async def refresh_figure(figure: str) -> dict[str, Any]:
        if figure not in ALL_FIGURES:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown figure: {figure}",
            )
        compute.refresh_one(figure)
        return {"status": "ok", "launched": [figure]}

    @app.post("/api/verify")
    async def verify() -> dict[str, Any]:
        launched = compute.refresh_verification()
        return {"status": "ok", "launched": launched}

    # ── Figures API ───────────────────────────────────────

    @app.get("/api/figures/{name}")
    async def get_figure(name: str) -> Any:
        if name not in ALL_FIGURES:
            raise HTTPException(status_code=404, detail=f"Unknown figure: {name}")

        fig_path = state.figures_dir / f"{name}.json"
        if not fig_path.exists():
            raise HTTPException(status_code=404, detail=f"Figure not yet computed: {name}")

        return json.loads(fig_path.read_text())

    # ── Baseline API ──────────────────────────────────────

    @app.post("/api/baseline")
    async def set_baseline() -> dict[str, Any]:
        result = state.set_baseline()
        broadcaster.broadcast({"type": "baseline_set", "summary": result.get("summary", {})})
        return result

    @app.delete("/api/baseline")
    async def clear_baseline() -> dict[str, str]:
        state.clear_baseline()
        broadcaster.broadcast({"type": "baseline_cleared"})
        return {"status": "ok"}

    # ── SSE stream ────────────────────────────────────────

    @app.get("/api/events")
    async def sse_stream() -> StreamingResponse:
        q = broadcaster.subscribe()

        async def event_generator():
            try:
                while True:
                    data = await q.get()
                    yield f"data: {json.dumps(data)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Static files / SPA ────────────────────────────────

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
