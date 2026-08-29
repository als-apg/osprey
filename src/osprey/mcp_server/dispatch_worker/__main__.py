"""Entry point for ``python -m osprey.mcp_server.dispatch_worker``."""

from __future__ import annotations

import os

import uvicorn

from osprey.port_layout import default_port, resolve_port_base


def main() -> None:
    """Run the dispatch worker FastAPI app via uvicorn.

    ``DISPATCH_WORKER_PORT`` is set by the compose service for every deployed
    worker, so the fallback below is only reached by a worker started by hand.
    It is the first port of the deployment's worker band — worker 1 — read from
    the project config this process can see, never from the layout's default
    base: a hand-started worker on a host running two deployments must not bind
    the other one's port.
    """
    from osprey.utils.logger import configure_logging
    from osprey.utils.workspace import load_osprey_config

    configure_logging()

    port = os.environ.get("DISPATCH_WORKER_PORT") or default_port(
        "worker", 1, base=resolve_port_base(load_osprey_config())
    )

    uvicorn.run(
        "osprey.mcp_server.dispatch_worker.dispatch_api:app",
        host=os.environ.get("DISPATCH_WORKER_BIND", "0.0.0.0"),
        port=int(port),
    )


if __name__ == "__main__":
    main()
