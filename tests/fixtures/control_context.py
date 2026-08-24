"""A controls server context wired to a connector-host manager, for tests.

``ControlSystemContext.initialize()`` reads a deployment's ``config.yml`` off
disk and builds the workspace singletons. Every suite that exercises the target
switch is downstream of that, so each one fills the same private fields by hand
instead — and three suites filling them separately is three places to update
when the context grows a field. It is spelled once, here.
"""

from __future__ import annotations

from typing import Any


def context_for(manager: Any) -> Any:
    """A server context wired to *manager*, without touching a real config.yml.

    Fills exactly what ``initialize()`` would have filled for the connectors
    these suites reach: the manager itself, the config it was built from, and
    the ``control_system`` / ``archiver`` entries the tools resolve through.
    """
    from osprey.mcp_server.control_system.server_context import (
        ConnectorEntry,
        ControlSystemContext,
    )

    context = ControlSystemContext()
    context._config = manager._config
    context._connector_hosts = manager
    context._connectors["control_system"] = ConnectorEntry(
        config=manager._config.control_system, connector_type="control_system"
    )
    context._connectors["archiver"] = ConnectorEntry(
        config=manager._config.archiver, connector_type="archiver"
    )
    return context
