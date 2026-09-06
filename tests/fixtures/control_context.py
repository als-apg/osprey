"""A controls server context wired to a connector-host manager, for tests.

``ControlSystemContext.initialize()`` reads a deployment's ``config.yml`` off
disk and builds the workspace singletons. Every suite that exercises the target
switch is downstream of that, so each one fills the same private fields by hand
instead — and three suites filling them separately is three places to update
when the context grows a field. It is spelled once, here.

The helpers below are the rest of what a controls server's lifespan does and a
test does not get for free: the control-context record it claims, the record
move that mints a generation, and the endpoint sweep the switch gate refuses a
destination for the want of. The reconcile loop that carries this server onto a
record somebody else moved is a fixture, in ``tests/va/e2e/conftest.py``.
"""

from __future__ import annotations

from dataclasses import replace
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
    claim_the_record(manager)
    return context


def claim_the_record(manager: Any) -> Any:
    """Claim the control-context record for the target *manager* is serving.

    A server with no record refuses every target switch, so a stand-in for the
    controls server's startup has to claim one the way it does. Claimed at the
    manager's target rather than at the deployment baseline: production writes
    the record first and brings the first child up on it, and these suites start
    the manager first, so a baseline claim would leave the record and the
    connector host naming different machines.

    ``claim_control_context`` follows a live owner rather than taking the record
    from it, so a suite whose web terminal owns the record keeps that owner.
    """
    from osprey.mcp_server.control_system.server_context import claim_control_context

    return claim_control_context(baseline=manager.active_target())


async def move_the_deployment(manager: Any, target: str) -> dict[str, Any]:
    """Move the deployment to *target* the way the record's owner moves it.

    The owner mints the next generation into the record and every server is
    then brought into line with it. ``ConnectorHostManager.switch`` mints
    nothing — it is handed a generation or it keeps the one it has — so a test
    that switches through it alone moves this server's child without moving the
    deployment, and leaves the record naming a target nobody is serving.

    Returns:
        What :meth:`ConnectorHostManager.reconcile` reported.
    """
    from osprey_connectors import control_context

    record = control_context.read_record()
    assert record is not None, "no control-context record to move — claim_the_record first"
    generation = record.generation + 1
    control_context.write_record(replace(record, target=target, generation=generation))
    control_context.invalidate_cache()
    return await manager.reconcile(target, generation)


async def publish_reachability(config: Any) -> None:
    """Publish one endpoint sweep, the way the prober's own loop publishes it.

    The switch gate refuses a destination that no live controls server has
    measured, and the prober that measures it runs in the server's lifespan
    rather than in a test. ``sweep_once`` only fills the prober's cache, so the
    loop's publish step is driven here too — that file is what the gate reads.
    """
    from osprey.mcp_server.control_system import target_state
    from osprey.mcp_server.control_system.endpoint_prober import EndpointProber

    prober = EndpointProber(config)
    await prober.sweep_once()
    target_state.publish_reachability(prober.reachability_rows())
