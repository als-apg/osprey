"""CLI command for the OSPREY Theme Lab.

Provides ``osprey theme-lab``: serves the design-system static directory on a
local port and opens the Theme Lab page, an interactive workbench for building
and previewing OSPREY themes against the live design tokens.

The command is intentionally project-independent — the Theme Lab reads only the
design system that ships inside the wheel, so it runs from any directory without
a ``config.yml``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from fastapi import FastAPI

#: Page the browser is pointed at, relative to the server root. Served by the
#: ``/design-system`` mount that ``configure_interface_app`` always registers.
THEME_LAB_PATH = "/design-system/theme-lab.html"


def _design_system_static_dir() -> Path:
    """Return the packaged design-system static directory.

    Resolved from the ``osprey.interfaces.design_system`` package itself rather
    than from the source tree layout, so an installed wheel and an editable
    checkout both find the same assets.
    """
    from osprey.interfaces import design_system

    return Path(design_system.__file__).parent / "static"


def create_app() -> FastAPI:
    """Build the FastAPI app that serves the Theme Lab.

    The design-system directory is passed as the interface's own ``static_dir``
    so its assets are reachable under ``/static`` as well as under the
    ``/design-system`` mount that :func:`configure_interface_app` registers
    automatically for every interface.

    Importable and callable with no server running — the Playwright suite and
    the CLI both build the app through this one factory.
    """
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from osprey.interfaces._app_setup import configure_interface_app

    static_dir = _design_system_static_dir()
    if not static_dir.is_dir():
        raise FileNotFoundError(
            f"design-system static directory not found: {static_dir} "
            "(the OSPREY installation is incomplete)"
        )

    app = FastAPI(title="OSPREY Theme Lab")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(THEME_LAB_PATH)

    configure_interface_app(app, static_dir=static_dir)

    return app


@click.command("theme-lab")
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port to serve on (default: an unused port chosen automatically).",
)
@click.option("--no-browser", is_flag=True, help="Do not open a browser window.")
def theme_lab(port: int | None, no_browser: bool) -> None:
    """Serve the OSPREY Theme Lab in a browser.

    Starts a local server for the design system and opens the Theme Lab, where
    themes can be built and previewed against the live design tokens. The URL is
    printed so it can be opened by hand if the browser does not appear.

    Example:

    \b
        osprey theme-lab                   # Serve on a free port, open a browser
        osprey theme-lab --port 9000       # Serve on a specific port
        osprey theme-lab --no-browser      # Print the URL only
    """
    import uvicorn

    from osprey.interfaces._serving import free_port

    try:
        app = create_app()
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    serve_port = port if port is not None else free_port()
    url = f"http://127.0.0.1:{serve_port}{THEME_LAB_PATH}"

    if not no_browser:
        # A browser that refuses to open (headless host, no DISPLAY) must not
        # take the server down with it — the printed URL below is still usable.
        try:
            from osprey.interfaces.web_terminal.app import _open_browser_when_ready

            _open_browser_when_ready(url)
        except Exception as exc:
            click.echo(f"WARNING: could not open a browser ({exc}).", err=True)

    click.echo(f"Theme Lab: {url}")
    click.echo("Press Ctrl+C to stop\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=serve_port, log_level="warning")
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
