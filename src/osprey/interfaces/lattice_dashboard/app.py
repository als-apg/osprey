"""Lattice Dashboard — FastAPI application.

Serves the dashboard SPA, REST API for lattice state management,
and SSE stream for live figure updates.

The lattice comes from the render's simulator view. One task resolves the
selection off the request path: at startup, on ``POST /api/models/select``,
and when a request sees the view or the selected deck change. A request never
loads a deck itself; it answers the current selection, ``loading`` while a
resolve runs.

Usage::

    from osprey.interfaces.lattice_dashboard.app import create_app
    app = create_app()  # resolves the deployment's agent-data root and render
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.lattice_dashboard.catalog import (
    ModelCatalog,
    catalog_sources,
    read_catalog,
)
from osprey.interfaces.lattice_dashboard.compute import ComputeManager
from osprey.interfaces.lattice_dashboard.schema import (
    BaselineResponse,
    DataResponse,
    FigureNotCurrent,
    FigureResponse,
    FigureUnavailable,
    HealthResponse,
    LaunchResponse,
    ModelEntry,
    SelectionModel,
    SettingsResetResponse,
    SettingsResponse,
    StateResponse,
    StatusResponse,
)
from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    SINGLE_PASS_UNAVAILABLE,
    LatticeState,
    Selection,
    capabilities_for,
    describe_deck,
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
    """Manages per-client asyncio.Queue instances for SSE push.

    Called on the event loop only.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[dict]] = []

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=64)
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        with contextlib.suppress(ValueError):
            self._queues.remove(q)

    def broadcast(self, data: dict) -> None:
        for q in self._queues:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(data)


# ── Response documentation ────────────────────────────────

#: The body FastAPI gives an ``HTTPException``.
_DETAIL: dict[str, Any] = {"description": "Not found"}

#: A figure route's answers other than 200.
_FIGURE_REFUSALS: dict[int | str, dict[str, Any]] = {
    404: {"model": FigureNotCurrent, "description": "No figure for the inputs on screen"},
    409: {"model": FigureUnavailable, "description": "The selected model cannot draw it"},
}


# ── Request models ────────────────────────────────────────


class ParamRequest(BaseModel):
    family: str
    value: float


class SettingsRequest(BaseModel):
    settings: dict[str, dict[str, Any]]


class SelectModelRequest(BaseModel):
    name: str


# ── Selection ─────────────────────────────────────────────


def _file_signature(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def resolve_selection(catalog: ModelCatalog, wanted: str | None) -> Selection:
    """Resolve the model *wanted*, or the catalog's default when None.

    Loads the deck, so it is called off the event loop. Never raises: a model
    the engine or the deck stops on resolves to ``failed`` with the reason,
    and a model the catalog no longer lists resolves to ``none``, never to
    the deck it had before.
    """
    model = catalog.default() if wanted is None else catalog.get(wanted)
    if model is None:
        return Selection(status="none")
    if model.prepared is None:
        return Selection(model=model.name, status="failed", error=model.error)
    try:
        digest = hashlib.sha256(model.deck.read_bytes()).hexdigest()
        families, summary = describe_deck(model.deck)
    except Exception as exc:
        logger.exception("Failed to load the deck of model %s", model.name)
        return Selection(model=model.name, status="failed", error=f"{type(exc).__name__}: {exc}")
    return Selection(
        model=model.name,
        status="ready",
        deck=model.deck,
        deck_sha256=digest,
        prepared=model.prepared,
        capabilities=capabilities_for(model.prepared.solve),
        families=families,
        summary=summary,
    )


class _Resolver:
    """Keeps the state's selection on the render's view, one resolve at a time.

    Args:
        state: The dashboard state.
        compute: The figure workers; a changed selection recomputes its fast
            figures.
        broadcaster: Told after each resolve that the state changed.
        render_root: The render to read the simulator view from, or None.
    """

    def __init__(
        self,
        state: LatticeState,
        compute: ComputeManager,
        broadcaster: _SSEBroadcaster,
        render_root: Path | None,
    ) -> None:
        self._state = state
        self._compute = compute
        self._broadcaster = broadcaster
        self._render_root = render_root
        self._lock = asyncio.Lock()
        self._catalog_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._again = False
        self._select = False
        self._catalog: ModelCatalog | None = None
        self._catalog_signature: tuple[Any, ...] | None = None
        self._deck_signature: tuple[int, int] | None = None
        self._resolved = False

    @property
    def catalog(self) -> ModelCatalog | None:
        """The catalog last read, or None until one has been."""
        return self._catalog

    def _view_signature(self) -> tuple[Any, ...] | None:
        root = self._render_root
        return None if root is None else tuple(map(_file_signature, catalog_sources(root)))

    def check(self) -> None:
        """Schedule a resolve when the view or the selected deck changed since the last one.

        A change seen while a resolve runs is checked again once it ends.
        """
        if self._task is not None and not self._task.done():
            return
        if (
            not self._resolved
            or self._view_signature() != self._catalog_signature
            or _file_signature(self._state.selection.deck) != self._deck_signature
        ):
            self.schedule()

    def schedule(self, *, select: str | None = None) -> None:
        """Mark the selection loading and run a resolve on the event loop.

        Args:
            select: The model the operator just picked; its what-if inputs
                are reset even when it is the model already selected.
        """
        if select is not None:
            self._select = True
            self._state.selection = Selection(model=select, status="loading")
        elif self._state.selection.status != "loading":
            self._state.selection = dataclasses.replace(self._state.selection, status="loading")
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())
        else:
            self._again = True

    async def current_catalog(self) -> ModelCatalog:
        """Return the view's catalog, read off the loop when the view changed.

        Waits only for another catalog read, never for a running resolve.
        """
        async with self._catalog_lock:
            signature = self._view_signature()
            if self._catalog is None or signature != self._catalog_signature:
                self._catalog = await asyncio.to_thread(read_catalog, self._render_root)
                self._catalog_signature = signature
            return self._catalog

    async def stop(self) -> None:
        """Cancel a running resolve."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _wanted(self, catalog: ModelCatalog) -> str | None:
        """The operator's pick; with none, the model drawn last while the build still lists it."""
        wanted = self._state.wanted()
        if wanted is not None:
            return wanted
        last = self._state.last_model()
        return last if last is not None and catalog.get(last) is not None else None

    async def _run(self) -> None:
        while True:
            self._again = False
            select, self._select = self._select, False
            try:
                catalog = await self.current_catalog()
                async with self._lock:
                    selection = await asyncio.to_thread(
                        resolve_selection, catalog, self._wanted(catalog)
                    )
            except Exception as exc:
                logger.exception("Failed to resolve the selected model")
                selection = Selection(status="failed", error=f"{type(exc).__name__}: {exc}")
            if self._again:
                continue
            self._deck_signature = _file_signature(selection.deck)
            self._resolved = True
            reset = self._state.adopt(selection, reset=select)
            self._broadcaster.broadcast({"type": "state_updated"})
            if reset:
                self._compute.refresh_fast()
            return


# ── App factory ───────────────────────────────────────────


def create_app(workspace_root: Path | None = None, render_root: Path | None = None) -> FastAPI:
    """Create the Lattice Dashboard FastAPI application.

    Constructing the app loads no deck and starts no worker: the selection is
    resolved once the app starts. Every worker is put down when it shuts
    down.

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
    if render_root is None:
        render_root = Path(p).parent if (p := default_config_path()) else None

    state = LatticeState(ws_root / "lattice")
    broadcaster = _SSEBroadcaster()
    compute = ComputeManager(state, broadcaster)
    resolver = _Resolver(
        state, compute, broadcaster, Path(render_root) if render_root is not None else None
    )

    def selection_payload(selection: Selection) -> dict[str, Any]:
        capabilities = selection.capabilities
        return {
            "model": selection.model,
            "deck_sha256": selection.deck_sha256,
            "status": selection.status,
            "error": selection.error,
            "capabilities": {
                "figures": list(capabilities.figures),
                "fast_figures": list(capabilities.fast_figures),
                "verify": capabilities.verify,
            },
        }

    def state_payload() -> dict[str, Any]:
        selection = state.selection
        return {
            "selection": selection_payload(selection),
            "families": selection.families,
            "overrides": state.overrides,
            "summary": state.summary(),
            "figures": {name: compute.figure_status(name) for name in ALL_FIGURES},
            "baseline": state.get_baseline(),
            "settings": state.get_settings(),
            "notice": None if resolver.catalog is None else resolver.catalog.notice(),
        }

    def unavailable(name: str) -> JSONResponse | None:
        """The refusal for a figure the selected model cannot draw, or None."""
        selection = state.selection
        if selection.ready and name not in selection.capabilities.figures:
            return JSONResponse(
                status_code=409,
                content={"status": "unavailable", "reason": SINGLE_PASS_UNAVAILABLE},
            )
        return None

    def known(name: str) -> None:
        if name not in ALL_FIGURES:
            raise HTTPException(status_code=404, detail=f"Unknown figure: {name}")

    def current_figure(name: str) -> dict[str, Any] | JSONResponse:
        """Return figure *name*'s stored payload, or the response saying why there is none."""
        known(name)
        refusal = unavailable(name)
        if refusal is not None:
            return refusal
        status = compute.figure_status(name)
        payload = state.read_figure(name) if status["status"] == "ready" else None
        if payload is None:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_computed" if status["status"] == "ready" else status["status"],
                    "key": status["key"],
                    "error": status["error"],
                },
            )
        return payload

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        resolver.schedule()
        # No worker outlives the app.
        try:
            yield
        finally:
            await resolver.stop()
            await compute.stop_all()

    app = FastAPI(
        title="Lattice Dashboard",
        description="Live lattice visualization dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.lattice = state
    app.state.compute = compute

    # ── Health ────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "lattice_dashboard"}

    # ── State API ─────────────────────────────────────────

    @app.get("/api/state", response_model=StateResponse)
    async def get_state() -> dict[str, Any]:
        resolver.check()
        return state_payload()

    # ── Models API ────────────────────────────────────────

    @app.get("/api/models", response_model=list[ModelEntry])
    async def list_models() -> list[dict[str, Any]]:
        resolver.check()
        selected = state.selection.model
        return [
            {
                "name": model.name,
                "served": model.served,
                "solve": model.solve,
                "selected": model.name == selected,
                "error": model.error,
            }
            for model in (resolver.catalog.models if resolver.catalog is not None else ())
        ]

    @app.post("/api/models/select", response_model=SelectionModel, responses={404: _DETAIL})
    async def select_model(body: SelectModelRequest) -> dict[str, Any]:
        catalog = await resolver.current_catalog()
        if catalog.get(body.name) is None:
            raise HTTPException(status_code=404, detail=f"Unknown model: {body.name}")
        state.set_wanted(body.name)
        resolver.schedule(select=body.name)
        return selection_payload(state.selection)

    @app.post("/api/state/param", response_model=StateResponse, responses={404: _DETAIL})
    async def set_param(body: ParamRequest) -> dict[str, Any]:
        if body.family not in state.selection.families:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown family: {body.family}",
            )

        state.set_param(body.family, body.value)
        broadcaster.broadcast({"type": "state_updated"})
        return state_payload()

    # ── Refresh API ───────────────────────────────────────

    @app.post("/api/refresh", response_model=LaunchResponse)
    async def refresh_fast() -> dict[str, Any]:
        return {"status": "ok", "launched": compute.refresh_fast()}

    @app.post(
        "/api/refresh/{figure}",
        response_model=LaunchResponse,
        responses={404: _DETAIL, 409: {"model": FigureUnavailable}},
    )
    async def refresh_figure(figure: str) -> dict[str, Any] | JSONResponse:
        known(figure)
        refusal = unavailable(figure)
        if refusal is not None:
            return refusal
        launched = [figure] if compute.refresh_one(figure) else []
        return {"status": "ok", "launched": launched}

    @app.post(
        "/api/verify", response_model=LaunchResponse, responses={409: {"model": FigureUnavailable}}
    )
    async def verify() -> dict[str, Any] | JSONResponse:
        selection = state.selection
        if selection.ready and not selection.capabilities.verify:
            return JSONResponse(
                status_code=409,
                content={"status": "unavailable", "reason": SINGLE_PASS_UNAVAILABLE},
            )
        return {"status": "ok", "launched": compute.refresh_verification()}

    # ── Figures API ───────────────────────────────────────

    @app.get("/api/figures/{name}", response_model=FigureResponse, responses=_FIGURE_REFUSALS)
    async def get_figure(name: str) -> dict[str, Any] | JSONResponse:
        payload = current_figure(name)
        if isinstance(payload, JSONResponse):
            return payload
        builder = FIGURE_BUILDERS[name]
        figure = figure_to_dict(builder(payload["data"]))
        return {"status": "ready", "key": payload["key"], "figure": figure}

    @app.get("/api/data/{name}", response_model=DataResponse, responses=_FIGURE_REFUSALS)
    async def get_data(name: str) -> dict[str, Any] | JSONResponse:
        payload = current_figure(name)
        if isinstance(payload, JSONResponse):
            return payload
        return {"status": "ready", "key": payload["key"], "data": payload["data"]}

    # ── Baseline API ──────────────────────────────────────

    @app.post("/api/baseline", response_model=BaselineResponse)
    async def set_baseline() -> dict[str, Any]:
        result = state.set_baseline()
        broadcaster.broadcast({"type": "baseline_set", "summary": result.get("summary", {})})
        return result

    @app.delete("/api/baseline", response_model=StatusResponse)
    async def clear_baseline() -> dict[str, str]:
        state.clear_baseline()
        broadcaster.broadcast({"type": "baseline_cleared"})
        return {"status": "ok"}

    # ── Settings API ──────────────────────────────────────

    @app.get("/api/settings", response_model=SettingsResponse)
    async def get_settings() -> dict[str, Any]:
        return state.get_settings()

    @app.put("/api/settings", response_model=SettingsResponse)
    async def update_settings(body: SettingsRequest) -> dict[str, Any]:
        result = state.update_settings(body.settings)
        broadcaster.broadcast({"type": "settings_updated", "settings": result})
        return result

    @app.delete("/api/settings", response_model=SettingsResetResponse)
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

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {})

    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
