"""CLI command that serves the history page.

Provides ``osprey history``: the entry of the ``history`` service the
deployment renders when its web terminals sign people in. The page is reached
only through the history page's origin, whose nginx
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

    Reads the agent record through its service on the viewer's behalf. It
    expects the environment the deployment renders for it: the record's
    address and the page's login, and the gate secret nginx presents on every
    request it forwards.

    Example:

    \b
        osprey history --host 0.0.0.0 --port 8000
    """
    import uvicorn

    from osprey.history.app import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
