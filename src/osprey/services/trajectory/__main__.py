"""Entry point for ``python -m osprey.services.trajectory``."""

from __future__ import annotations

import math
import os
import socket
from collections.abc import Mapping
from typing import Any

import uvicorn
from fastapi import FastAPI

from osprey.port_layout import default_port, resolve_port_base

#: The address the service binds; loopback unless the deployment says otherwise.
BIND_ENV_VAR = "TRAJECTORY_BIND"
DEFAULT_BIND = "127.0.0.1"


def service_port(config: Mapping[str, Any]) -> int:
    """The port of ``services.trajectory.port``, or the layout's trajectory slot.

    The fallback is the slot on the deployment's own port base, read from the
    project config this process can see, so a hand-started service on a host
    running two deployments does not bind the other one's port.
    """
    services = config.get("services")
    block = services.get("trajectory") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if port is not None and str(port).strip():
        return int(port)
    return default_port("trajectory", base=resolve_port_base(config))


class TrajectoryServer(uvicorn.Server):
    """A uvicorn server that stops the service's work before it drains connections.

    On the way out it awaits :meth:`Sessions.shutdown` first, so every state
    stream has ended and every queued action failed before uvicorn waits for
    open connections, then shuts down as uvicorn does. The graceful drain is
    bounded by the service's stop budget, read once the lifespan has started it.
    """

    def __init__(self, config: uvicorn.Config, app: FastAPI) -> None:
        super().__init__(config)
        self.app = app

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        sessions = getattr(self.app.state, "sessions", None)
        if sessions is not None:
            self.config.timeout_graceful_shutdown = math.ceil(sessions.budgets().stop_budget_s)

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        sessions = getattr(self.app.state, "sessions", None)
        if sessions is not None:
            await sessions.shutdown()
        await super().shutdown(sockets=sockets)


def main() -> None:
    """Serve the trajectory service's app via :class:`TrajectoryServer`."""
    from osprey.services.trajectory.app import create_app
    from osprey_connectors.logger import configure_logging
    from osprey_connectors.workspace import load_osprey_config

    configure_logging()
    app = create_app()
    config = uvicorn.Config(
        app,
        host=os.environ.get(BIND_ENV_VAR, "").strip() or DEFAULT_BIND,
        port=service_port(load_osprey_config()),
    )
    TrajectoryServer(config, app).run()


if __name__ == "__main__":
    main()
