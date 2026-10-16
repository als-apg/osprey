"""CLI command that serves the history page.

Provides ``osprey history``: the entry of the ``history`` service the
deployment renders when it runs the telemetry store and its web terminals sign
people in. The page is reached only through the readers' origin, whose nginx
asks the auth sidecar who the browser is and forwards the answer with the
page's gate secret; this command takes everything it needs from the
environment the compose file sets, and acts on no deployment repo.
"""

from __future__ import annotations

import click


@click.command("history")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to listen on (the container listens on every interface).",
)
@click.option("--port", "-p", type=int, default=8000, show_default=True, help="Port to serve on.")
def history(host: str, port: int) -> None:
    """Serve the history page (the history service's entry).

    Reads the telemetry store as the store's ``history`` account and runs every
    query as the viewer's own store view user. It expects the environment the
    deployment renders for it: the store's address and the account's password,
    and the gate secret nginx presents on every request it forwards.

    Example:

    \b
        osprey history --host 0.0.0.0 --port 8000
    """
    import uvicorn

    from osprey.history.app import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
