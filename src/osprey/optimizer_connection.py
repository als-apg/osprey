"""Single source of truth for resolving the optimizer service's base URL.

Two holders resolve it and must agree: the optimizer MCP server
(``osprey.mcp_server.optimizer.server_context``), which dials it, and the Reach
Contract (``osprey.deployment.reach``), whose ``reach`` health probe knocks on
what that server would dial. A top-level leaf module, like
``osprey.bluesky_bridge_connection``: it imports nothing from
``osprey.mcp_server`` or ``osprey.deployment``, and loads the config lazily.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from osprey.port_layout import default_port, resolve_port_base

#: The port the optimizer publishes at the layout's default base.
DEFAULT_OPTIMIZER_PORT = default_port("optimizer")

#: Names the optimizer's base URL outright, overriding the rendered config.
OPTIMIZER_URL_ENV_VAR = "OPTIMIZER_URL"


def optimizer_url_from_config(config: Mapping[str, Any] | None) -> str:
    """The optimizer base URL a loaded ``config.yml`` implies.

    ``services.optimizer.port`` on loopback — the port the deployment publishes
    the service on, written by the build and projected into attached renders —
    else the layout's optimizer slot at this config's ``deployment.port_base``.
    """
    services = (config or {}).get("services")
    block = services.get("optimizer") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if not port:
        port = default_port("optimizer", base=resolve_port_base(config))
    return f"http://127.0.0.1:{port}"


def resolve_optimizer_url() -> str:
    """The optimizer base URL, trailing slash stripped.

    :data:`OPTIMIZER_URL_ENV_VAR` wins when set and non-empty; otherwise the
    loaded ``config.yml`` through :func:`optimizer_url_from_config`.
    """
    full = os.environ.get(OPTIMIZER_URL_ENV_VAR)
    if full:
        return full.rstrip("/")

    from osprey_connectors.workspace import load_osprey_config

    return optimizer_url_from_config(load_osprey_config())
