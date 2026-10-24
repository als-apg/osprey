"""CLI group for the agent-record service.

Provides ``osprey agent-record serve``: the entry of the ``agent-record``
service every deployment runs. It takes everything it needs from the rendered
``config.yml`` and the environment the compose file sets (the verifier and the
secret set, the spool, record and transcript locations), and acts on no
deployment repo.
"""

from __future__ import annotations

import click


@click.group("agent-record")
def agent_record_group() -> None:
    """The agent record: one store of every agent's transcript, audit and run facts."""


@agent_record_group.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to listen on (the container listens on every interface).",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port to serve on.  [default: the configured agent_record.port]",
)
def serve(host: str, port: int | None) -> None:
    """Serve agent-record: OTLP ingest, the read API and source health.

    Ingests the transcript roots and the spool into the configured backend and
    answers reads for the credentials the deployment minted. It expects the
    environment the deployment renders for it: the verifier and the secret set,
    and the spool, record and transcript locations.

    Example:

    \b
        osprey agent-record serve --host 0.0.0.0
    """
    import uvicorn

    from osprey.agent_record.service import create_app

    app = create_app()
    if port is None:
        port = app.state.settings.record.port
    uvicorn.run(app, host=host, port=port, log_level="warning")
