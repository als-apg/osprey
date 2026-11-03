"""The dispatch dashboard as the browser suites in this directory stand it up.

The dashboard has no standalone FastAPI app of its own in production: it is
rendered HTML mounted into the event-dispatcher server. A minimal real app that
mounts the real design-system static dir and serves the real
``render_dashboard_html()`` output is the honest way to drive it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import osprey.interfaces.design_system as design_system_pkg

DESIGN_SYSTEM_STATIC_DIR = Path(design_system_pkg.__file__).parent / "static"


def mount_design_system(app: FastAPI) -> None:
    """Serve the real design-system static dir where the dashboard's links expect it."""
    app.mount(
        "/design-system", StaticFiles(directory=DESIGN_SYSTEM_STATIC_DIR), name="design-system"
    )


def create_dispatch_dashboard_app(
    state: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    *,
    channel_strip_prefix: str = "SR:",
    history_url: str = "",
) -> FastAPI:
    """Serve the real dashboard beside the real design system.

    Args:
        state: What ``/dashboard/state`` answers, or a callable building it per
            request. ``None`` serves no state route at all.
        channel_strip_prefix: Passed to ``render_dashboard_html()``.
        history_url: Passed to ``render_dashboard_html()``.
    """
    from osprey.dispatch.dashboard import render_dashboard_html

    app = FastAPI()
    mount_design_system(app)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return str(
            render_dashboard_html(
                facility_name="Test Facility",
                channel_strip_prefix=channel_strip_prefix,
                history_url=history_url,
            )
        )

    if state is not None:

        @app.get("/dashboard/state")
        async def dashboard_state() -> dict[str, Any]:
            return state() if callable(state) else state

    return app
