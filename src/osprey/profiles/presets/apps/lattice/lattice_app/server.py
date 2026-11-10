"""The application's backend: the panel, and the routes the panel and the agent both read.

Every route is a read. A result is computed the first time anyone asks for it and kept
(:mod:`lattice_app.jobs`); what runs, on which decks and with which settings is the
deployment's choice (:mod:`lattice_app.config`), not the caller's, which is what keeps
one result shared by every terminal.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from lattice_app import APP_NAME, EXPORT_ROUTE, MAX_WAIT_S, RESULT_ROUTE, STATE_ROUTE
from lattice_app.catalog import ANALYSES
from lattice_app.config import Model, Settings, load_settings
from lattice_app.export import FORMATS, render
from lattice_app.jobs import Jobs, Request, Runner, build_request, run_worker
from osprey.apps.sdk import configure_app

#: Where the deployment hands this application its rendered ``apps.<name>.config``.
APP_CONFIG_ENV = "OSPREY_APP_CONFIG"
#: Where the deployment lists the application's state directories, colon-separated.
STATE_DIRS_ENV = "OSPREY_APP_STATE_DIRS"
#: What the state says when the deployment offers no deck.
NO_MODELS = (
    f"no lattice model is configured: list decks under apps.{APP_NAME}.config.models "
    "in the deployment's profile"
)


def panel_dir() -> Path:
    """The panel's files: inside the installed package, else beside it in the source tree."""
    package = Path(__file__).resolve().parent
    installed = package / "panel"
    return installed if installed.is_dir() else package.parent / "panel"


def read_app_config() -> dict[str, Any]:
    """The application's config the deployment rendered; empty when it rendered none."""
    path = os.environ.get(APP_CONFIG_ENV)
    if not path or not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def cache_dir() -> Path:
    """Where results are kept: the declared state directory, else a temporary one."""
    declared = [entry for entry in os.environ.get(STATE_DIRS_ENV, "").split(":") if entry]
    root = Path(declared[0]) if declared else Path(tempfile.gettempdir()) / f"osprey-{APP_NAME}"
    return root / "cache"


def _energy(model: Model, asked: float | None) -> float | None:
    """The energy a request means: the asked one if the model offers it, else its first."""
    if asked is None:
        return model.energies_gev[0] if model.energies_gev else None
    for offered in model.energies_gev:
        if abs(offered - asked) <= 1.0e-9 * max(1.0, abs(offered)):
            return offered
    offered_text = ", ".join(f"{energy:g}" for energy in model.energies_gev) or "none"
    raise HTTPException(
        400, f"model {model.name} offers no energy {asked:g} GeV; it offers: {offered_text}"
    )


def _describe(model: Model) -> dict[str, Any]:
    return {
        "name": model.name,
        "label": model.label,
        "engine": model.engine.name,
        "engine_settings": dict(model.engine_settings),
        "deck": model.deck.name,
        "energies_gev": list(model.energies_gev),
        "analyses": model.offered(),
    }


def create_app(
    settings: Settings | None = None,
    *,
    cache: Path | None = None,
    runner: Runner = run_worker,
) -> FastAPI:
    """Build the backend.

    Args:
        settings: The deployment's choices; read from the application's config when
            omitted.
        cache: Where results are kept; the declared state directory when omitted.
        runner: Runs one worker.
    """
    error: str | None = None
    if settings is None:
        try:
            settings = load_settings(read_app_config())
        except ValueError as exc:
            settings, error = Settings(models=()), f"the application's config is not usable: {exc}"
    chosen: Settings = settings
    if error is None and not chosen.models:
        error = NO_MODELS

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        jobs = Jobs(cache_dir() if cache is None else cache, chosen.max_workers, runner)
        app.state.jobs = jobs
        if chosen.precompute:
            for model in chosen.models:
                for energy in model.energies_gev or (None,):
                    for analysis in model.offered():
                        # An unreadable deck is reported when its result is asked for.
                        with contextlib.suppress(ValueError):
                            jobs.ensure(build_request(model, analysis, energy))
        yield
        await jobs.close()

    app = FastAPI(title=APP_NAME, lifespan=lifespan)

    def find(model_name: str | None) -> Model:
        if not chosen.models:
            raise HTTPException(409, error or NO_MODELS)
        if model_name is None:
            return chosen.models[0]
        if (model := chosen.model(model_name)) is None:
            names = ", ".join(model.name for model in chosen.models)
            raise HTTPException(404, f"there is no model {model_name}; the models: {names}")
        return model

    def request_for(model: Model, analysis: str, energy: float | None) -> Request:
        if analysis not in ANALYSES:
            raise HTTPException(
                404, f"there is no analysis {analysis}; the analyses: {', '.join(ANALYSES)}"
            )
        if analysis not in model.offered():
            offered = ", ".join(model.offered()) or "none"
            raise HTTPException(
                400, f"model {model.name} does not offer {analysis}; it offers: {offered}"
            )
        try:
            return build_request(model, analysis, energy)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    def setup_of(model: Model, analysis: str) -> dict[str, Any]:
        setup = model.setups[analysis]
        return {"settings": dict(setup.settings), "deadline_s": setup.deadline_s}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(panel_dir() / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(STATE_ROUTE)
    async def state(model: str | None = None, energy_gev: float | None = None) -> dict[str, Any]:
        """Everything the panel shows but the tables: models, analyses, and each result's status."""
        answer: dict[str, Any] = {
            "error": error,
            "models": [_describe(model) for model in chosen.models],
            "analyses": [
                {"name": analysis.name, "label": analysis.label} for analysis in ANALYSES.values()
            ],
            "selected": None,
            "results": {},
        }
        if not chosen.models:
            return answer
        selected = find(model)
        energy = _energy(selected, energy_gev)
        answer["selected"] = {"model": selected.name, "energy_gev": energy}
        jobs: Jobs = app.state.jobs
        for analysis in selected.offered():
            try:
                status = jobs.status(build_request(selected, analysis, energy).key)
            except ValueError as exc:
                status = {"status": "failed", "error": str(exc)}
            answer["results"][analysis] = {**setup_of(selected, analysis), **status}
        return answer

    @app.get(RESULT_ROUTE)
    async def result(
        analysis: str,
        model: str | None = None,
        energy_gev: float | None = None,
        wait_s: float = 0.0,
        retry: bool = False,
    ) -> dict[str, Any]:
        """One analysis's result, computed if nobody has asked for it yet.

        Waits up to ``wait_s`` for a running computation; the answer's ``status`` says
        whether the tables are in it.
        """
        selected = find(model)
        energy = _energy(selected, energy_gev)
        request = request_for(selected, analysis, energy)
        jobs: Jobs = app.state.jobs
        jobs.ensure(request, retry=retry)
        await jobs.wait(request.key, min(max(wait_s, 0.0), MAX_WAIT_S))
        answer = {
            "model": selected.name,
            "analysis": analysis,
            "energy_gev": energy,
            "engine": selected.engine.name,
            "engine_settings": dict(selected.engine_settings),
            **setup_of(selected, analysis),
            **jobs.status(request.key),
        }
        if answer["status"] == "ready" and (document := jobs.result(request.key)) is not None:
            answer["tables"] = document["tables"]
        return answer

    @app.get(EXPORT_ROUTE)
    async def export(
        analysis: str,
        model: str | None = None,
        energy_gev: float | None = None,
        format: str = "csv",
        table: str | None = None,
    ) -> PlainTextResponse:
        """A ready result's table as a CSV or SDDS file."""
        if format not in FORMATS:
            raise HTTPException(400, f"there is no format {format}; the formats: csv, sdds")
        selected = find(model)
        energy = _energy(selected, energy_gev)
        request = request_for(selected, analysis, energy)
        jobs: Jobs = app.state.jobs
        document = jobs.result(request.key)
        if document is None:
            status = jobs.status(request.key)["status"]
            raise HTTPException(
                409, f"{analysis} of {selected.name} is {status}, not ready; read its result first"
            )
        name = table or next(iter(ANALYSES[analysis].tables))
        if name not in document["tables"]:
            tables = ", ".join(document["tables"])
            raise HTTPException(404, f"{analysis} has no table {name}; its tables: {tables}")
        at_energy = "" if energy is None else f"-{energy:g}GeV"
        filename = f"{selected.name}-{analysis}-{name}{at_energy}.{format}"
        return PlainTextResponse(
            render(document, name, format),
            media_type=FORMATS[format],
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    configure_app(app, static_dir=panel_dir())
    return app


def main() -> None:
    """Serve the backend on the port and address the deployment names."""
    uvicorn.run(
        create_app(),
        host=os.environ.get("OSPREY_APP_BIND", "127.0.0.1"),
        port=int(os.environ["OSPREY_APP_PORT"]),
    )


if __name__ == "__main__":
    main()
