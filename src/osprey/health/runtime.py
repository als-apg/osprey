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

import asyncio
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any

from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from osprey.connectors.archiver.base import ArchiverConnector

logger = logging.getLogger("osprey.health.runtime")

#: The subject the health runtime speaks under in the shared baseline-pinned
#: wording. Spelled once here so the row and any future health refusal cannot
#: drift apart.
HEALTH_SUBJECT = "HealthRuntime"

#: Name and category of the informational row. ``control_system`` is not a
#: health *category* anyone can run — no probe lives there — which is the point:
#: the row is a banner about the whole report, not a check that passed or failed.
BASELINE_ROW_NAME = "control_system.target"
BASELINE_ROW_CATEGORY = "control_system"


class HealthRuntime:
    """Async context manager owning at most one lazily-created connector.

    The connector is constructed on the first :meth:`get_connector` call and
    cached; subsequent calls return the same instance. On context exit (or an
    explicit :meth:`shutdown`) the connector is disconnected exactly once, and
    only if one was actually constructed. Teardown is best-effort: a
    ``disconnect()`` that raises is swallowed so a failing connector can never
    mask the suite's own result.

    Construction is serialized. Health checks run concurrently — probes within
    a category are awaited together, and the categories themselves run under a
    single ``gather`` — so several probes can reach an accessor while the
    connector is still ``None``. Without a guard each of them would see the
    empty slot, each would build a connector, and every assignment but the last
    would orphan one: an unreferenced Channel Access client that :meth:`shutdown`
    never disconnects, because it only ever knew about the instance that won the
    race. Both accessors therefore take a single :class:`asyncio.Lock` and
    re-test the cache inside it, so exactly one caller constructs and every
    other waiter returns that same instance. The ``closed`` refusal stays
    outside the lock: a runtime that has already torn its connectors down must
    say so immediately rather than queue behind an in-flight construction.

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
        #: Guards lazy construction in both accessors (see the class docstring).
        #: Held only across construction, never across teardown.
        self._lock = asyncio.Lock()

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

        Concurrent callers are serialized, so two probes that both find the slot
        empty still produce one connector between them; the closed check runs
        before the lock, so a closed runtime refuses at once.
        """
        if self._closed:
            raise RuntimeError(
                "HealthRuntime is closed; get_connector() cannot reconstruct "
                "a connector after shutdown()"
            )
        async with self._lock:
            if self._connector is None:
                from osprey.connectors.factory import (
                    ConnectorFactory,
                    register_builtin_connectors,
                )

                register_builtin_connectors()  # idempotent; must run before create
                self._connector = await ConnectorFactory.create_control_system_connector(
                    self._config
                )
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

        Construction is serialized on the same lock as :meth:`get_connector`, so
        concurrent archiver probes build one connector between them rather than
        one apiece.

        Args:
            archiver_config: The ``archiver`` config block passed straight to
                :meth:`ConnectorFactory.create_archiver_connector`.
        """
        if self._closed:
            raise RuntimeError(
                "HealthRuntime is closed; get_archiver() cannot reconstruct "
                "an archiver connector after shutdown()"
            )
        async with self._lock:
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

    @staticmethod
    def baseline_pinned_row() -> CheckResult | None:
        """The informational row naming both targets, or ``None`` on the baseline.

        The health suite reports on the deployment *as configured*: its
        connectors are built from the config section fixed at construction, and
        a session-level control-system target switch does not move them. That is
        the never-swap ruling, and this row does not change it — it makes it
        visible. While a session is switched away, a reader would otherwise take
        an ``ok`` row about the baseline target as an ``ok`` row about the target
        they are working on.

        The sentence comes from
        :func:`osprey.mcp_server.control_system.target_banner.baseline_pinned_line`,
        the same helper the Phoebus tools render their label from, so the two
        pinned holders cannot word the same fact two ways. ``None`` on the
        baseline keeps an unswitched report byte-identical to what it was before
        this row existed.

        Emitted as :data:`~osprey.health.models.Status.SKIP`: it is not a check,
        so it must not count as one that passed, and skip is the one status that
        leaves :attr:`~osprey.health.models.CheckReport.exit_code` alone.

        Returns:
            The row while the session is switched, otherwise ``None``. Never
            raises — a banner that failed to render must not fail a health run.
        """
        try:
            from osprey.mcp_server.control_system.target_banner import baseline_pinned_line

            line = baseline_pinned_line(HEALTH_SUBJECT)
        except Exception:  # noqa: BLE001 - a label can never cost a health run
            logger.debug("Could not resolve the baseline-pinned row (ignored)", exc_info=True)
            return None
        if line is None:
            return None
        return CheckResult(BASELINE_ROW_NAME, BASELINE_ROW_CATEGORY, Status.SKIP, line)

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
