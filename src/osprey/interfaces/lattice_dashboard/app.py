"""Lattice Dashboard — FastAPI application.

Serves the dashboard SPA, REST API for lattice state management,
and SSE stream for live figure updates.

Usage::

    from osprey.interfaces.lattice_dashboard.app import create_app
    app = create_app(workspace_root=Path("./_agent_data"))
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import ALL_FIGURES, DEFAULT_SETTINGS, LatticeState
from osprey.interfaces.lattice_dashboard.workers._base import figure_to_dict
from osprey.interfaces.lattice_dashboard.workers.chromaticity import (
    build_figure as build_chromaticity,
)
from osprey.interfaces.lattice_dashboard.workers.da import build_figure as build_da
from osprey.interfaces.lattice_dashboard.workers.footprint import (
    build_figure as build_footprint,
)
from osprey.interfaces.lattice_dashboard.workers.lma import build_figure as build_lma
from osprey.interfaces.lattice_dashboard.workers.optics import build_figure as build_optics
from osprey.interfaces.lattice_dashboard.workers.resonance import (
    build_figure as build_resonance,
)
from osprey.interfaces.vendor import vendor_url

logger = logging.getLogger("osprey.lattice_dashboard")

STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url


# ── Figure adapters ──────────────────────────────────────
# Each adapter unpacks a raw data dict and calls the worker's
# build_figure(), converting lists → numpy arrays as needed.


def _build_optics(raw: dict) -> go.Figure:
    baseline = None
    if raw.get("baseline"):
        b = raw["baseline"]
        baseline = (
            np.array(b["s_pos"]),
            np.array(b["beta_x"]),
            np.array(b["beta_y"]),
            np.array(b["eta_x"]),
        )
    return build_optics(
        np.array(raw["s_pos"]),
        np.array(raw["beta_x"]),
        np.array(raw["beta_y"]),
        np.array(raw["eta_x"]),
        baseline,
    )


def _build_chromaticity(raw: dict) -> go.Figure:
    baseline = None
    if raw.get("baseline"):
        b = raw["baseline"]
        baseline = (np.array(b["dp"]), np.array(b["nux"]), np.array(b["nuy"]))
    return build_chromaticity(
        np.array(raw["dp"]),
        np.array(raw["nux"]),
        np.array(raw["nuy"]),
        baseline,
    )


def _build_footprint(raw: dict) -> go.Figure:
    baseline = None
    if raw.get("baseline"):
        b = raw["baseline"]
        baseline = (np.array(b["nux"]), np.array(b["nuy"]), np.array(b["amps"]))
    design_tune = tuple(raw["design_tune"]) if raw.get("design_tune") else None
    baseline_tune = tuple(raw["baseline_tune"]) if raw.get("baseline_tune") else None
    n_amp = raw.get("n_amp", 10)
    return build_footprint(
        np.array(raw["nux"]),
        np.array(raw["nuy"]),
        np.array(raw["amps"]),
        np.array(raw.get("diffusion", [])),
        design_tune=design_tune,
        baseline_tune=baseline_tune,
        baseline=baseline,
        n_total=n_amp * n_amp,
    )


def _build_resonance(raw: dict) -> go.Figure:
    return build_resonance(
        raw["nux"],
        raw["nuy"],
        raw.get("baseline_nux"),
        raw.get("baseline_nuy"),
    )


def _build_da_figure(raw: dict) -> go.Figure:
    baseline = None
    if raw.get("baseline"):
        b = raw["baseline"]
        baseline = (np.array(b["da_x"]), np.array(b["da_y"]), b["area_mm2"])
    return build_da(
        np.array(raw["da_x"]),
        np.array(raw["da_y"]),
        raw["area_mm2"],
        nturns=raw.get("nturns", 512),
        baseline=baseline,
    )


def _build_lma_figure(raw: dict) -> go.Figure:
    baseline = None
    if raw.get("baseline"):
        b = raw["baseline"]
        baseline = (np.array(b["s_pos"]), np.array(b["dp_plus"]), np.array(b["dp_minus"]))
    return build_lma(
        np.array(raw["s_pos"]),
        np.array(raw["dp_plus"]),
        np.array(raw["dp_minus"]),
        raw.get("lattice_elements", []),
        baseline=baseline,
        n_sectors=raw.get("n_sectors", 1),
    )


FIGURE_BUILDERS: dict[str, Any] = {
    "optics": _build_optics,
    "chromaticity": _build_chromaticity,
    "footprint": _build_footprint,
    "resonance": _build_resonance,
    "da": _build_da_figure,
    "lma": _build_lma_figure,
}


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


class SettingsRequest(BaseModel):
    settings: dict[str, dict[str, Any]]


# ── App factory ───────────────────────────────────────────


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the Lattice Dashboard FastAPI application.

    Args:
        workspace_root: Workspace root (e.g. ``./_agent_data``).
            The lattice state lives under ``<workspace>/lattice/``.
    """
    ws_root = Path(workspace_root) if workspace_root else Path("./_agent_data")
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
        s = state.load()
        if "settings" not in s:
            s["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        return s

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

        raw = json.loads(fig_path.read_text())
        builder = FIGURE_BUILDERS.get(name)
        if builder is None:
            return raw  # fallback for unknown figure types
        fig = builder(raw)
        return figure_to_dict(fig)

    @app.get("/api/data/{name}")
    async def get_data(name: str) -> Any:
        if name not in ALL_FIGURES:
            raise HTTPException(status_code=404, detail=f"Unknown figure: {name}")

        fig_path = state.figures_dir / f"{name}.json"
        if not fig_path.exists():
            raise HTTPException(status_code=404, detail=f"Data not yet computed: {name}")

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

    # ── Settings API ──────────────────────────────────────

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return state.get_settings()

    @app.put("/api/settings")
    async def update_settings(body: SettingsRequest) -> dict[str, Any]:
        result = state.update_settings(body.settings)
        broadcaster.broadcast({"type": "settings_updated", "settings": result})
        return result

    @app.delete("/api/settings")
    async def reset_settings() -> dict[str, Any]:
        result = state.reset_settings()
        broadcaster.broadcast({"type": "settings_updated", "settings": result})
        return {"status": "ok", "settings": result}

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
    async def root(request: Request):
        return templates.TemplateResponse(request, "index.html", {})

    # Mount shared fonts before /static (Starlette matches in declaration order)
    SHARED_FONTS_DIR = Path(__file__).parent.parent / "shared_fonts"
    if SHARED_FONTS_DIR.exists():
        app.mount("/static/fonts", StaticFiles(directory=SHARED_FONTS_DIR), name="shared-fonts")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
