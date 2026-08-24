"""CLI commands for the OSPREY Artifact Gallery.

Provides `osprey artifacts web` to launch the gallery server manually.
"""

import os

import click

from . import output


@click.group("artifacts")
def artifacts():
    """Artifact Gallery commands.

    Manage the OSPREY Artifact Gallery — a local web gallery that displays
    interactive plots, tables, and other outputs produced by Claude during
    analysis sessions.
    """


@artifacts.command("web")
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port to run on (default: OSPREY_ARTIFACT_SERVER_PORT, then config, then 8086)",
)
@click.option(
    "--host", "-h", default=None, help="Host to bind to (default: from config or 127.0.0.1)"
)
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def web(port: int | None, host: str | None, reload: bool) -> None:
    """Launch the Artifact Gallery web interface.

    Starts a FastAPI server that serves the artifact gallery UI.
    Artifacts are created by Claude via save_artifact() in execute
    or the artifact_register MCP tool.

    Example:

    \b
        osprey artifacts web                    # Start on localhost:8086
        osprey artifacts web --port 9000        # Custom port
        osprey artifacts web --host 0.0.0.0     # Bind to all interfaces
        osprey artifacts web --reload           # Development mode
    """
    from osprey.registry.web import resolve_web_server_address

    # Explicit flags win; otherwise the framework's shared derivation, which
    # applies the OSPREY_ARTIFACT_SERVER_PORT override the deployment may set.
    default_host, default_port = resolve_web_server_address("artifact")
    host = host or default_host
    port = port or default_port

    output.report(f"Starting OSPREY Artifact Gallery on http://{host}:{port}")
    output.note("Press Ctrl+C to stop")
    output.report("")

    try:
        if reload:
            import uvicorn

            from osprey.interfaces.common_middleware import WEB_PORT_ENV
            from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, mint_and_announce

            # Under --reload the ChangeReload worker uvicorn spawns re-imports
            # the factory in a FRESH process, so the operator secret must be
            # minted HERE, in this parent, before the spawn — ``mint_and_announce``
            # re-publishes it in ``os.environ[OSPREY_TERMINAL_SECRET]`` and the
            # worker inherits and pops it at its own ``create_app``. Minting in
            # the factory instead would give each reload a different secret and
            # invalidate the announced URL. The non-reload branch does not mint
            # here: ``run_server`` mints in-process (it is the serving process).
            # Publish the settled port before the app is built, here and in the
            # worker this spawns. Cookies ignore ports, so two OSPREY servers on
            # one host share an origin as far as the browser is concerned; the
            # port is what keeps their session cookies from overwriting each
            # other, and ``session_cookie_name()`` reads it from here.
            os.environ[WEB_PORT_ENV] = str(port)

            announce = not (os.environ.get(OPERATOR_SECRET_ENV) or "").strip()
            login_url = mint_and_announce(host, port)
            if announce:
                output.report(f"Open: {login_url}")

            uvicorn.run(
                "osprey.interfaces.artifacts.app:create_app",
                factory=True,
                host=host,
                port=port,
                reload=reload,
                log_level="info",
            )
        else:
            from osprey.interfaces.artifacts import run_server

            run_server(host=host, port=port)
    except KeyboardInterrupt:
        output.report("")
        output.report("Shutting down...")
