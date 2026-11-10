"""Single source of truth for resolving the optimizer service's base URL.

Three holders resolve it and must agree: the optimizer MCP server
(``osprey.mcp_server.optimizer.server_context``), which dials it, the operator
panel's sidecar (``osprey.interfaces.optimizer_web``), which relays to it, and
the Reach Contract (``osprey.deployment.reach``), whose ``reach`` health probe
knocks on what that server would dial. A top-level leaf module, like
``osprey.bluesky_bridge_connection``: it imports nothing from
``osprey.mcp_server`` or ``osprey.deployment``.
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

#: Holds the deployment secret that arms the optimizer's gated routes (start,
#: resume, ...), the same name the service reads it under. Only the operator
#: panel's sidecar (``osprey.interfaces.optimizer_web``) is handed it; the MCP
#: server never is, and the agent child paths strip every ``*_LAUNCH_TOKEN``
#: (``osprey.utils.sensitive_env``), so an agent can draft a run but not start one.
OPTIMIZER_LAUNCH_TOKEN_ENV_VAR = "OPTIMIZER_LAUNCH_TOKEN"  # noqa: S105 -- names an environment variable, holds no secret

#: Seconds a client of the service waits for an answer. Longer than the
#: service's own deadlines on a machine read (the connector pool's 60 s, the
#: environment's 75 s backstop), so the service's typed error arrives before
#: the client gives up. Shared by the MCP server and the panel's sidecar.
OPTIMIZER_TIMEOUT_S = 80.0


def optimizer_url(config: Mapping[str, Any] | None) -> str:
    """The optimizer base URL, trailing slash stripped.

    :data:`OPTIMIZER_URL_ENV_VAR` when set and non-empty; else
    ``services.optimizer.port`` on loopback (the port the deployment publishes,
    written by the build and projected into attached renders); else the
    layout's optimizer slot at this config's ``deployment.port_base``.
    """
    full = os.environ.get(OPTIMIZER_URL_ENV_VAR)
    if full:
        return full.rstrip("/")
    services = (config or {}).get("services")
    block = services.get("optimizer") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if not port:
        port = default_port("optimizer", base=resolve_port_base(config))
    return f"http://127.0.0.1:{port}"
