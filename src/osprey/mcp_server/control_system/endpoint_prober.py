"""Continuous, honest reachability rows for both control targets (FR-2).

The roster answers "may this session be pointed there" from config alone, which
is a question about configuration rather than about the machine. This module
answers the other half — *is anything listening* — and it answers it in the
background, so that the roster tool call itself stays side-effect-free: a read
tool that opens sockets would make reporting a target's state indistinguishable
from acting on it.

A background task re-derives both targets' endpoints and re-probes every
``control_system.target_switch.probe_interval_s`` seconds (default 30), starting
with an immediate sweep so the roster has rows before the first interval
elapses. Each row records the gateway it was taken against, so a row can never
be read as a claim about an endpoint it was not measured at.

What a row does and does not prove
----------------------------------
The probe is a plain TCP connect and nothing else. No CA library is loaded here,
no channel is read, and no EPICS traffic is generated.

That is why a gateway configured with ``use_name_server: false`` reports
``not_applicable`` rather than a status, **and is not probed at all**: that mode
puts the gateway in ``EPICS_CA_ADDR_LIST``, and CA search on an address list is
UDP. A TCP connection proves a socket is listening; it proves nothing about
whether CA search will find anything, and a successful TCP handshake reported as
"reachable" would be a claim this probe cannot support. On a stock EPICS
deployment — address-list mode — the live target therefore has no liveness row
at all, and ``spawn_probe`` (a real channel read in the connector-host child) is
its only real liveness evidence. That is a smaller claim than the roster could
make; it is the one it can actually keep.

For a ``use_name_server: true`` gateway the name server *is* a TCP service, so a
connect attempt is exactly the right instrument, and ``ok``/``unreachable`` mean
what they say about that socket.

Staleness is a read-time verdict
--------------------------------
:meth:`EndpointProber.snapshot` computes staleness when the roster reads, never
when the prober writes: a row does not become stale by being rewritten, it
becomes stale by not being. A row whose age exceeds
:data:`STALENESS_INTERVALS` × the probe interval reports ``stale`` while keeping
what it last measured in ``last_status``, so a stalled prober is visible without
destroying the last real observation.

Ages are measured on :func:`time.monotonic`, which no clock adjustment can move;
``probed_at`` carries the wall-clock ISO timestamp for display only. A row is
never called stale because someone changed the system clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from osprey.mcp_server.control_system.target_eligibility import (
    MODE_NAME_SERVER,
    Endpoint,
    derive_endpoints,
)
from osprey_connectors.types import CONTROL_TARGETS

logger = logging.getLogger(__name__)

# -- Config key, spelled once ----------------------------------------------

#: How often the background sweep runs.
PROBE_INTERVAL_KEY = "control_system.target_switch.probe_interval_s"
DEFAULT_PROBE_INTERVAL_S = 30.0

#: How long one TCP connect attempt may take. A constructor argument rather
#: than a config key: it is a property of the instrument, not of the deployment,
#: and a gateway that needs more than a couple of seconds to accept a socket is
#: already the answer the row is trying to report.
DEFAULT_CONNECT_TIMEOUT_S = 2.0

#: A row older than this many probe intervals reports :data:`STATUS_STALE`.
STALENESS_INTERVALS = 3

# -- Row statuses -----------------------------------------------------------

#: The gateway accepted a TCP connection.
STATUS_OK = "ok"
#: The gateway refused, timed out, or could not be resolved.
STATUS_UNREACHABLE = "unreachable"
#: Address-list mode: CA search is UDP, so TCP cannot answer the question.
STATUS_NOT_APPLICABLE = "not_applicable"
#: Read-time only: no sweep has refreshed this row in too long.
STATUS_STALE = "stale"

ADDR_LIST_DETAIL = (
    "This gateway is configured with use_name_server: false, so it is reached "
    "through EPICS_CA_ADDR_LIST and CA search runs over UDP. A TCP connection "
    "would prove a socket is listening, not that CA search resolves, so no "
    "connection is attempted and no reachability is claimed."
)


@dataclass(frozen=True)
class EndpointRow:
    """One role's reachability, as last measured.

    ``probed_monotonic`` is the clock staleness is computed on; ``probed_at`` is
    the same instant in wall-clock terms, for a human reading the roster.
    """

    endpoint_tcp: str
    gateway: str
    probed_at: str
    probed_monotonic: float
    detail: str = ""

    def as_dict(self, *, stale: bool = False) -> dict[str, Any]:
        """The row as the roster reports it, optionally aged out to ``stale``.

        ``last_status`` always carries what was actually measured, so a stale
        row still says what it saw the last time it managed to look.
        """
        return {
            "endpoint_tcp": STATUS_STALE if stale else self.endpoint_tcp,
            "last_status": self.endpoint_tcp,
            "gateway": self.gateway,
            "probed_at": self.probed_at,
            "detail": self.detail,
        }


class EndpointProber:
    """Background TCP reachability for every configured gateway of both targets.

    Constructed with the rendered config and nothing else, so a server lifespan
    can own one with ``prober = EndpointProber(config)``, ``await
    prober.start()`` at startup and ``await prober.stop()`` at shutdown.

    The cache is keyed ``{target: {role: EndpointRow}}``. A target whose
    endpoints cannot be derived — unresolvable, or configured with no gateways —
    is *absent* from the cache rather than present with an error row: the prober
    reports reachability, and "this deployment has no such target" is the
    roster's eligibility answer, not a reachability observation. Derivation is
    redone every sweep, so a target that becomes derivable later fills in
    without a restart.
    """

    def __init__(
        self,
        config: Any,
        *,
        targets: tuple[str, ...] | list[str] | None = None,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        interval_s: float | None = None,
        monotonic: Any = time.monotonic,
    ) -> None:
        """Create a prober.

        Args:
            config: The full rendered config mapping, as
                :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
                takes it.
            targets: The session targets to probe. Defaults to every target the
                connectors package defines.
            connect_timeout_s: Per-endpoint TCP connect timeout.
            interval_s: Sweep interval, overriding :data:`PROBE_INTERVAL_KEY`.
            monotonic: The monotonic clock, injectable so a test can age rows
                without sleeping. Never a wall clock: staleness must survive a
                clock adjustment.
        """
        self._config = config
        self._targets = tuple(targets) if targets is not None else tuple(CONTROL_TARGETS)
        self._connect_timeout_s = float(connect_timeout_s)
        self._interval_s = (
            _positive_float(interval_s, DEFAULT_PROBE_INTERVAL_S)
            if interval_s is not None
            else _configured_interval(config)
        )
        self._monotonic = monotonic
        self._cache: dict[str, dict[str, EndpointRow]] = {}
        self._task: asyncio.Task[None] | None = None
        #: Set once the first sweep has finished, so a caller that needs rows
        #: (a test, or a roster call racing startup) can await them rather than
        #: guess how long a sweep takes.
        self.first_sweep_done = asyncio.Event()

    # -- Introspection ------------------------------------------------------

    @property
    def probe_interval_s(self) -> float:
        """The sweep interval in force, after config reading and defaulting."""
        return self._interval_s

    @property
    def staleness_threshold_s(self) -> float:
        """The age beyond which :meth:`snapshot` reports a row as stale."""
        return self._interval_s * STALENESS_INTERVALS

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the sweep task. Idempotent; returns without blocking on a sweep.

        The first sweep is the task's first action rather than something awaited
        here, so server startup is never delayed by an unreachable gateway's
        connect timeout. Await :attr:`first_sweep_done` when rows are needed.
        """
        if self.running:
            return
        self.first_sweep_done = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="control-endpoint-prober")

    async def stop(self) -> None:
        """Cancel the sweep task and wait for it to finish. Idempotent."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        """Sweep immediately, then once per interval until cancelled."""
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive; a sweep swallows its own
                logger.exception("Endpoint probe sweep failed; continuing")
            finally:
                self.first_sweep_done.set()
            await asyncio.sleep(self._interval_s)

    # -- Probing ------------------------------------------------------------

    async def sweep_once(self) -> None:
        """Re-derive and re-probe every target's endpoints, once.

        Public so a caller can force a refresh deterministically (the switch
        path, or a test) without waiting for the loop.
        """
        for target in self._targets:
            try:
                derivation = derive_endpoints(self._config, target)
            except Exception as exc:
                # An underivable target has no endpoints to report on. Dropping
                # it leaves the roster's eligibility answer to explain why,
                # rather than inventing a reachability row for an endpoint this
                # deployment never described.
                logger.debug("No endpoints derivable for target %r: %s", target, exc)
                self._cache.pop(target, None)
                continue

            roles = list(derivation.endpoints)
            if not roles:
                self._cache.pop(target, None)
                continue

            rows = await asyncio.gather(
                *(self._probe(derivation.endpoints[role]) for role in roles)
            )
            self._cache[target] = dict(zip(roles, rows, strict=True))

    async def _probe(self, endpoint: Endpoint) -> EndpointRow:
        """One endpoint's row. Connects only in name-server mode."""
        gateway = f"{endpoint.host}:{endpoint.port}"

        if endpoint.mode != MODE_NAME_SERVER:
            return self._row(STATUS_NOT_APPLICABLE, gateway, ADDR_LIST_DETAIL)

        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(endpoint.host, endpoint.port),
                timeout=self._connect_timeout_s,
            )
        except TimeoutError:
            return self._row(
                STATUS_UNREACHABLE,
                gateway,
                f"No TCP connection to {gateway} within {self._connect_timeout_s}s.",
            )
        except (OSError, ValueError, TypeError) as exc:
            return self._row(
                STATUS_UNREACHABLE,
                gateway,
                f"{type(exc).__name__}: {exc}",
            )

        writer.close()
        with contextlib.suppress(OSError, asyncio.TimeoutError, TimeoutError):
            await writer.wait_closed()
        return self._row(STATUS_OK, gateway, "")

    def _row(self, status: str, gateway: str, detail: str) -> EndpointRow:
        return EndpointRow(
            endpoint_tcp=status,
            gateway=gateway,
            probed_at=datetime.now(UTC).isoformat(),
            probed_monotonic=float(self._monotonic()),
            detail=detail,
        )

    # -- Reading ------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The current rows, aged out at read time. Synchronous and side-effect-free.

        Every dict returned is freshly built, so a caller may edit the result
        (the roster does, folding these rows into its per-role endpoint table)
        without reaching into the cache.

        A row older than :attr:`staleness_threshold_s` reports ``stale``, except
        a ``not_applicable`` row: that verdict comes from configuration, not
        from a measurement, so there is nothing about it to go out of date.
        """
        now = float(self._monotonic())
        threshold = self.staleness_threshold_s
        return {
            target: {
                role: row.as_dict(
                    stale=(
                        row.endpoint_tcp != STATUS_NOT_APPLICABLE
                        and (now - row.probed_monotonic) > threshold
                    )
                )
                for role, row in rows.items()
            }
            for target, rows in self._cache.items()
        }


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _positive_float(value: Any, default: float) -> float:
    """*value* as a positive float, or *default* when it is neither."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _configured_interval(config: Any) -> float:
    """:data:`PROBE_INTERVAL_KEY` from a rendered config, nested sections only."""
    section: Any = config
    for part in PROBE_INTERVAL_KEY.split(".")[:-1]:
        if not isinstance(section, dict):
            return DEFAULT_PROBE_INTERVAL_S
        section = section.get(part)
    if not isinstance(section, dict):
        return DEFAULT_PROBE_INTERVAL_S
    return _positive_float(
        section.get(PROBE_INTERVAL_KEY.rsplit(".", 1)[1]), DEFAULT_PROBE_INTERVAL_S
    )
