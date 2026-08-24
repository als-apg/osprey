"""CLI command for the OSPREY Theme Lab.

Provides ``osprey theme-lab``: serves the design-system static directory on a
local port and opens the Theme Lab page, an interactive workbench for building
and previewing OSPREY themes against the live design tokens.

The command is intentionally project-independent — the Theme Lab reads only the
design system that ships inside the wheel, so it runs from any directory without
a ``config.yml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from . import output

if TYPE_CHECKING:
    from fastapi import FastAPI

#: Page the browser is pointed at, relative to the server root. Served by the
#: ``/design-system`` mount that ``configure_interface_app`` always registers.
#:
#: **Not the page the login URL points at.** That mount is unauthenticated —
#: :func:`~osprey.interfaces.common_middleware.is_exempt_path` clears anything
#: under ``/design-system`` before a credential is looked at — so a ``?token=``
#: appended here is never read, never exchanged for a cookie, and the page loads
#: with no session behind it. The login URL points at ``/``, which the gate does
#: inspect; the root route below then redirects here, by which time the browser
#: is carrying the session cookie the exchange handed it.
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
        # Also the landing point of the ``?token=`` login URL: WebAuthMiddleware
        # answers that request itself (cookie + redirect to a clean ``/``), and
        # the browser's follow-up arrives here with a session, whereupon this
        # sends it on to the Theme Lab page.
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
    from osprey.interfaces.common_middleware import WEB_PORT_ENV
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, mint_and_announce

    # Capture whether an operator secret was already supplied BEFORE create_app
    # runs. This must precede construction: create_app -> configure_interface_app
    # resolves the credential holder and then closes the OSPREY_TERMINAL_SECRET
    # carrier (pops it out of os.environ), so reading it afterward would always
    # look absent and make the suppression gate below dead. A deployment that supplied the secret
    # reads as present here and stays silent; a secret this process mints reads
    # as absent and is announced.
    announce = not (os.environ.get(OPERATOR_SECRET_ENV) or "").strip()

    serve_port = port if port is not None else free_port()

    # Publish the settled port before anything builds the app. Cookies ignore
    # ports, so two OSPREY servers on this host share an origin as far as the
    # browser is concerned; the port is what keeps their session cookies from
    # overwriting each other, and ``session_cookie_name()`` reads it from here.
    os.environ[WEB_PORT_ENV] = str(serve_port)

    # Mint the operator secret in this CLI parent, which is also the process
    # that serves (direct-serve: uvicorn.run(app) below runs in-process, so the
    # app object built below shares this holder). ``mint_and_announce`` returns
    # the ``?token=`` login URL — the operator's only way past the auth
    # middleware — and settles the secret; ``announce`` (captured above) gates
    # the print so a deployment-supplied secret is not re-echoed here. The path
    # is the app root, not THEME_LAB_PATH: see the note on that constant.
    login_url = mint_and_announce("127.0.0.1", serve_port)

    try:
        app = create_app()
    except FileNotFoundError as exc:
        output.fail("Cannot start the Theme Lab", str(exc))
        sys.exit(1)

    if not no_browser:
        # A browser that refuses to open (headless host, no DISPLAY) must not
        # take the server down with it — the printed URL below is still usable.
        # Open the token URL: the gate trades the token for a session cookie and
        # redirects to ``/``, which then redirects to the Theme Lab. Opening
        # THEME_LAB_PATH directly would render the page with no session, and the
        # first credentialed request it made would be refused.
        try:
            from osprey.interfaces.web_terminal.app import _open_browser_when_ready

            _open_browser_when_ready(login_url)
        except Exception as exc:
            output.warn("Could not open a browser", str(exc))

    output.report(f"Starting the OSPREY Theme Lab on http://127.0.0.1:{serve_port}{THEME_LAB_PATH}")
    if announce:
        # The one line carrying the secret. Printed only when this process
        # minted it; an inherited secret was already announced upstream.
        output.report(f"Open: {login_url}")
    output.note("Press Ctrl+C to stop")
    output.report("")

    try:
        uvicorn.run(app, host="127.0.0.1", port=serve_port, log_level="warning")
    except KeyboardInterrupt:
        output.report("")
        output.report("Shutting down...")
