"""Health runtime — single owner of the control-system connector lifecycle.

`HealthRuntime` is an async context manager that lazily constructs at most one
control-system connector and disconnects it exactly once on exit. It is the sole
owner of the connector's lifecycle for a health-suite run: probes that need a
control-system connection (e.g. ``channel_read``) acquire it via
:meth:`HealthRuntime.get_connector`, which constructs the connector on first
call and caches it for the rest of the suite. A suite with no such probes never
triggers construction, so no Channel Access client is created — and no PV is
left for the garbage collector to finalize (a GC-finalized pyepics PV segfaults
libca).

The get/create/shutdown surface deliberately mirrors
:class:`osprey.mcp_server.control_system.server_context.ControlSystemContext`
and the construct-once/disconnect-once template in
``osprey.services.bluesky_bridge.app._lifespan`` so that P2 (FastAPI lifespan +
teardown hook) and P3 (per-process cache, ``server_context`` idiom) can reuse
this class without API changes.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

from osprey.connectors.control_system.base import ControlSystemConnector

logger = logging.getLogger("osprey.health.runtime")


class HealthRuntime:
    """Async context manager owning at most one lazily-created connector.

    The connector is constructed on the first :meth:`get_connector` call and
    cached; subsequent calls return the same instance. On context exit (or an
    explicit :meth:`shutdown`) the connector is disconnected exactly once, and
    only if one was actually constructed. Teardown is best-effort: a
    ``disconnect()`` that raises is swallowed so a failing connector can never
    mask the suite's own result.

    Args:
        control_system_config: The ``control_system`` config section passed
            straight to
            :meth:`ConnectorFactory.create_control_system_connector`.
    """

    def __init__(self, control_system_config: dict[str, Any]) -> None:
        self._config = control_system_config
        self._connector: ControlSystemConnector | None = None
        self._ever_constructed = False
        self._closed = False

    @property
    def ever_constructed(self) -> bool:
        """Whether a connector was ever successfully constructed.

        Becomes ``True`` at the first successful :meth:`get_connector` and
        stays ``True`` thereafter. Unlike ``_connector is None`` — which
        :meth:`shutdown` also produces — this records construction *history*,
        letting callers distinguish "never built a connector" from "built one
        and then tore it down".
        """
        return self._ever_constructed

    @property
    def closed(self) -> bool:
        """Whether :meth:`shutdown` has run (via explicit call or context exit).

        Once ``True``, :meth:`get_connector` refuses rather than reconstructing.
        """
        return self._closed

    async def __aenter__(self) -> HealthRuntime:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    async def get_connector(self) -> ControlSystemConnector:
        """Return the cached connector, constructing it on first call.

        The first call registers the built-in connector types (idempotent) and
        creates the control-system connector from the configured section. Later
        calls return the same cached instance. After :meth:`shutdown` the
        runtime is closed and this refuses with :class:`RuntimeError` rather
        than reconstructing a connector the suite already tore down.
        """
        if self._closed:
            raise RuntimeError(
                "HealthRuntime is closed; get_connector() cannot reconstruct "
                "a connector after shutdown()"
            )
        if self._connector is None:
            from osprey.connectors.factory import (
                ConnectorFactory,
                register_builtin_connectors,
            )

            register_builtin_connectors()  # idempotent; must run before create
            self._connector = await ConnectorFactory.create_control_system_connector(self._config)
            self._ever_constructed = True
            logger.info(
                "HealthRuntime: constructed control-system connector (%s)",
                type(self._connector).__name__,
            )
        return self._connector

    async def shutdown(self) -> None:
        """Disconnect the connector exactly once, iff one was constructed.

        A never-constructed runtime disconnects nothing but still marks the
        runtime closed. Disconnect exceptions are swallowed (best-effort
        teardown), and the cached instance is cleared so a repeated call
        disconnects nothing.
        """
        self._closed = True
        if self._connector is None:
            return
        try:
            await self._connector.disconnect()
        except Exception:
            logger.debug(
                "HealthRuntime: error disconnecting connector (ignored)",
                exc_info=True,
            )
        finally:
            self._connector = None
