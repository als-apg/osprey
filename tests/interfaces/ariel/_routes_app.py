"""ARIEL's API router on a bare app, serving a given search service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from osprey.interfaces.ariel.api import routes


def routes_app(service: Any) -> FastAPI:
    """A routes-only app serving ``service``.

    The lifespan resolves the config-panel tier flag; a routes-only app states it.
    """
    app = FastAPI()
    app.include_router(routes.router)
    app.state.ariel_service = service
    app.state.config_panel_enabled = True
    return app
