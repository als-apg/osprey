"""Health runtime — single owner of the control-system connector lifecycle.

`HealthRuntime` is an async context manager that lazily constructs at most one
control-system connector and at most one archiver connector, disconnecting each
exactly once on exit. It is the sole owner of both connectors' lifecycles for a
health-suite run: probes that need a control-system connection (e.g.
``channel_read``) acquire it via :meth:`HealthRuntime.get_connector`, and probes
that query the archiver (e.g. ``archiver_freshness``) acquire it via
:meth:`HealthRuntime.get_archiver`; each constructs its connector on first call
and caches it for the rest of the suite. A suite with no such probes never
triggers construction, so no Channel Access client is created — and no PV is
left for the garbage collector to finalize (a GC-finalized pyepics PV segfaults
libca).

The two accessors differ in how they source their config, reflecting the two
connectors' asymmetric roles. The control-system connector is the runtime's
raison d'être — its careful single-ownership exists precisely to keep the CA
client's lifecycle safe — so its config is fixed at construction. The archiver
connector is a secondary, HTTP-class resource (EPICS Archiver Appliance,
MongoDB, DOOCS — none Channel Access), so :meth:`get_archiver` takes its config
block per call, letting the probe honor an explicit per-run ``ctx.config`` over
the global singleton without threading archiver config through every
``HealthRuntime`` construction site.

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
from typing import TYPE_CHECKING, Any

from osprey.connectors.control_system.base import ControlSystemConnector

if TYPE_CHECKING:
    from osprey.connectors.archiver.base import ArchiverConnector

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
        self._archiver: ArchiverConnector | None = None
        self._archiver_ever_constructed = False
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
    def archiver_ever_constructed(self) -> bool:
        """Whether an archiver connector was ever successfully constructed.

        Tracked separately from :attr:`ever_constructed` (which records the
        control-system connector's history), so the archiver's presence never
        influences control-system-specific reconciliation such as the web
        sidecar's config-change restart notice.
        """
        return self._archiver_ever_constructed

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

    async def get_archiver(self, archiver_config: dict[str, Any]) -> ArchiverConnector:
        """Return the cached archiver connector, constructing it on first call.

        The first call registers the built-in connector types (idempotent) and
        creates the archiver connector from *archiver_config* — the ``archiver:``
        config block (``type`` plus the per-type sub-block), which the caller
        resolves with the correct precedence (an explicit per-run ``ctx.config``
        over the global singleton). ``ConnectorFactory.create_archiver_connector``
        connects on construction, so a first call that returns is proof the
        archiver was reachable. Later calls return the same cached instance and
        ignore *archiver_config* — a suite runs against one archiver.

        After :meth:`shutdown` the runtime is closed and this refuses with
        :class:`RuntimeError` rather than reconstructing a connector the suite
        already tore down.

        Args:
            archiver_config: The ``archiver`` config block passed straight to
                :meth:`ConnectorFactory.create_archiver_connector`.
        """
        if self._closed:
            raise RuntimeError(
                "HealthRuntime is closed; get_archiver() cannot reconstruct "
                "an archiver connector after shutdown()"
            )
        if self._archiver is None:
            from osprey.connectors.factory import (
                ConnectorFactory,
                register_builtin_connectors,
            )

            register_builtin_connectors()  # idempotent; must run before create
            self._archiver = await ConnectorFactory.create_archiver_connector(archiver_config)
            self._archiver_ever_constructed = True
            logger.info(
                "HealthRuntime: constructed archiver connector (%s)",
                type(self._archiver).__name__,
            )
        return self._archiver

    async def shutdown(self) -> None:
        """Disconnect both connectors exactly once, iff each was constructed.

        A never-constructed runtime disconnects nothing but still marks the
        runtime closed. Disconnect exceptions are swallowed (best-effort
        teardown), and each cached instance is cleared so a repeated call
        disconnects nothing.
        """
        self._closed = True
        await self._disconnect_one("connector", self._connector)
        self._connector = None
        await self._disconnect_one("archiver connector", self._archiver)
        self._archiver = None

    @staticmethod
    async def _disconnect_one(
        label: str,
        connector: ControlSystemConnector | ArchiverConnector | None,
    ) -> None:
        """Disconnect one connector best-effort, swallowing any exception.

        A ``None`` connector (never constructed, or already cleared) is a no-op;
        a ``disconnect()`` that raises is logged at debug and swallowed so a
        failing connector can never mask the suite's own result.
        """
        if connector is None:
            return
        try:
            await connector.disconnect()
        except Exception:
            logger.debug(
                "HealthRuntime: error disconnecting %s (ignored)",
                label,
                exc_info=True,
            )
