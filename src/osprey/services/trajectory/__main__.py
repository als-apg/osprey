"""Entry point for ``python -m osprey.services.trajectory``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import uvicorn

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


def main() -> None:
    """Serve the trajectory service's app via uvicorn."""
    from osprey.utils.logger import configure_logging
    from osprey_connectors.workspace import load_osprey_config

    configure_logging()
    uvicorn.run(
        "osprey.services.trajectory.app:create_app",
        factory=True,
        host=os.environ.get(BIND_ENV_VAR, "").strip() or DEFAULT_BIND,
        port=service_port(load_osprey_config()),
    )


if __name__ == "__main__":
    main()
