"""Lattice Dashboard — FastAPI application.

Serves the dashboard SPA, REST API for lattice state management,
and SSE stream for live figure updates.

The lattice comes from the render's simulator view: the dashboard loads the
first served deck-bearing model's deck on the first ``/api/state`` or
``/api/models`` request, and ``POST /api/models/select`` switches to another.

Usage::

    from osprey.interfaces.lattice_dashboard.app import create_app
    app = create_app()  # resolves the deployment's agent-data root and render
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.lattice_dashboard.catalog import (
    DashboardModel,
    ModelCatalog,
    catalog_sources,
    read_catalog,
)
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    DEFAULT_SETTINGS,
    SINGLE_PASS,
    SINGLE_PASS_UNAVAILABLE,
    LatticeState,
    fast_figures,
    figure_available,
)
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
from osprey.utils.config import default_config_path
from osprey.utils.workspace import resolve_shared_data_root

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


class ParamRequest(BaseModel):
    family: str
    value: float


class SettingsRequest(BaseModel):
    settings: dict[str, dict[str, Any]]


class SelectModelRequest(BaseModel):
    name: str


# ── Model loading ─────────────────────────────────────────


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class _ModelLoader:
    """Keeps the dashboard state on the selected model's deck.

    The catalog is re-read when ``served_models.json`` or ``facility.json``
    changes, and a deck's digest when the deck file changes, so a rebuilt
    render is picked up without re-reading it on every request.

    Args:
        state: The dashboard state.
        render_root: The render to read the simulator view from, or None.
    """

    def __init__(self, state: LatticeState, render_root: Path | None) -> None:
        self._state = state
        self._render_root = render_root
        self._lock = threading.Lock()
        self._catalog: ModelCatalog | None = None
        self._catalog_key: tuple[Any, ...] | None = None
        self._digests: dict[Path, tuple[tuple[int, int] | None, str]] = {}

    def catalog(self) -> ModelCatalog:
        """Return the render's switchable models."""
        root = self._render_root
        key = (
            None
            if root is None
            else tuple(_file_signature(source) for source in catalog_sources(root))
        )
        if self._catalog is None or key != self._catalog_key:
            self._catalog = read_catalog(root)
            self._catalog_key = key
        return self._catalog

    def _digest(self, deck: Path) -> str:
        signature = _file_signature(deck)
        cached = self._digests.get(deck)
        if cached is not None and cached[0] == signature:
            return cached[1]
        digest = hashlib.sha256(deck.read_bytes()).hexdigest()
        self._digests[deck] = (signature, digest)
        return digest

    def _load(self, model: DashboardModel, digest: str) -> dict[str, Any]:
        return self._state.initialize(
            str(model.deck),
            model=model.name,
            solve=model.solve,
            twiss_in=model.twiss_in,
            deck_sha256=digest,
        )

    def sync(self) -> bool:
        """Load the selected model's deck, else the default model's, when stale.

        Returns:
            True when the state was (re)initialised from a deck.
        """
        with self._lock:
            catalog = self.catalog()
            current = self._state.load()
            model = catalog.get(current.get("model") or "") or catalog.default()
            if model is None:
                return False
            digest = self._digest(model.deck)
            if current.get("model") == model.name and current.get("deck_sha256") == digest:
                return False
            self._load(model, digest)
            return True

    def select(self, name: str) -> DashboardModel | None:
        """Initialise the state from model *name*'s deck.

        Overrides and the old baseline are dropped; settings are kept.

        Returns:
            The model loaded, or None when the catalog has no model *name*.
        """
        with self._lock:
            model = self.catalog().get(name)
            if model is None:
                return None
            self._load(model, self._digest(model.deck))
            return model


# ── App factory ───────────────────────────────────────────


def create_app(workspace_root: Path | None = None, render_root: Path | None = None) -> FastAPI:
    """Create the Lattice Dashboard FastAPI application.

    Constructing the app loads no deck and starts no worker: the state is
    initialised on the first ``/api/state`` or ``/api/models`` request.

    Args:
        workspace_root: Agent-data root (e.g. ``<repo>/var/agent_data``). The
            lattice state lives under ``<workspace>/lattice/``. Omit it to
            resolve the deployment's configured root, which is what the
            framework launch path passes explicitly. Never default it to a
            cwd-relative directory: a caller standing anywhere but the repo root
            would get a fresh empty state rather than the running deployment's.
        render_root: The render whose ``data/simulator/`` view supplies the
            decks. Omit it to use the directory of the config this process
            loaded; with no config loaded there is no simulator view.
    """
    ws_root = Path(workspace_root) if workspace_root else resolve_shared_data_root()
    state_dir = ws_root / "lattice"
    if render_root is None:
        render_root = Path(p).parent if (p := default_config_path()) else None

    state = LatticeState(state_dir)
    broadcaster = _SSEBroadcaster()
    compute = ComputeManager(state, broadcaster)
    loader = _ModelLoader(state, Path(render_root) if render_root is not None else None)

    def sync_model() -> None:
        try:
            loaded = loader.sync()
        except Exception as exc:
            logger.exception("Failed to load the selected model's deck")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if loaded:
            broadcaster.broadcast({"type": "state_updated"})
            compute.refresh_fast()

    def state_payload() -> dict[str, Any]:
        s = state.load()
        if "settings" not in s:
            s["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        s.setdefault("model", None)
        s["fast_figures"] = list(fast_figures(s.get("solve")))
        s["notice"] = loader.catalog().notice()
        return s

    def refuse_unavailable(name: str) -> None:
        if not figure_available(name, state.load().get("solve")):
            raise HTTPException(status_code=409, detail=SINGLE_PASS_UNAVAILABLE)

    app = FastAPI(
        title="Lattice Dashboard",
        description="Live lattice visualization dashboard",
        version="1.0.0",
    )

    # ── Health ────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "lattice_dashboard"}

    # ── State API ─────────────────────────────────────────

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        sync_model()
        return state_payload()

    # ── Models API ────────────────────────────────────────

    @app.get("/api/models")
    async def list_models() -> list[dict[str, Any]]:
        sync_model()
        selected = state.load().get("model")
        return [
            {
                "name": model.name,
                "served": model.served,
                "solve": model.solve,
                "selected": model.name == selected,
            }
            for model in loader.catalog().models
        ]

    @app.post("/api/models/select")
    async def select_model(body: SelectModelRequest) -> dict[str, Any]:
        try:
            model = loader.select(body.name)
        except Exception as exc:
            logger.exception("Failed to load model %s", body.name)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if model is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {body.name}")

        broadcaster.broadcast({"type": "state_updated"})
        compute.refresh_fast()
        return state_payload()

    @app.post("/api/state/param")
    async def set_param(body: ParamRequest) -> dict[str, Any]:
        current = state.load()
        if body.family not in current.get("families", {}):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown family: {body.family}",
            )

        result = state.set_param(body.family, body.value)
        broadcaster.broadcast({"type": "state_updated"})
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
        refuse_unavailable(figure)
        compute.refresh_one(figure)
        return {"status": "ok", "launched": [figure]}

    @app.post("/api/verify")
    async def verify() -> dict[str, Any]:
        if state.load().get("solve") == SINGLE_PASS:
            raise HTTPException(status_code=409, detail=SINGLE_PASS_UNAVAILABLE)
        launched = compute.refresh_verification()
        return {"status": "ok", "launched": launched}

    # ── Figures API ───────────────────────────────────────

    @app.get("/api/figures/{name}")
    async def get_figure(name: str) -> Any:
        if name not in ALL_FIGURES:
            raise HTTPException(status_code=404, detail=f"Unknown figure: {name}")
        refuse_unavailable(name)

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
        refuse_unavailable(name)

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

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
